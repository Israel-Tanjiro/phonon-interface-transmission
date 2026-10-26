"""
phonon_interface_model.py  (copy of the user's solver; module docstring shortened,
code unchanged except for the added "Ta" entry in MATERIALS)

Every-Koos-Wolfe (PRB 29, 2190, 1984) critical-cone-channeling / loose-bonding model
of phonon transmission from an isotropic crystal into a metal film, with L, SV and SH
channels, Lambertian angular averaging and Debye weighting (1 L + 2 T).
Cross-checked against Kaplan, J. Low Temp. Phys. 37, 343 (1979) in kaplan_comparison().
See the PCE report (Appendix C) for the full derivation.
"""
import json
import numpy as np
_trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy >= 2.0 / older

MATERIALS = {
    "sapphire": dict(
        V_L=11.1e3, V_T=6.04e3, rho=3.986e3,
        source="Every, Koos & Wolfe, PRB 29, 2190 (1984) -- isotropic "
               "approximation to trigonal Al2O3 along the c-axis, their "
               "Fig. 10/12/13 captions."),
    "Cu": dict(
        V_L=4.70e3, V_T=2.26e3, rho=8.93e3,
        source="Every, Koos & Wolfe (1984), their Ref. 16."),
    "Al": dict(
        V_L=6.26e3, V_T=3.08e3, rho=2.70e3,
        source="Every, Koos & Wolfe (1984), their Ref. 16 (used in their "
               "Fig. 13)."),
    "Nb": dict(
        V_L=5068.0, V_T=2092.0, rho=8.570e3,
        source="Standard polycrystalline bulk longitudinal/transverse "
               "ultrasonic reference values (density 8570 kg/m^3). "
               "Cross-checked against single-crystal elastic constants "
               "C11~246, C12~134, C44~28 GPa via Voigt-Reuss-Hill "
               "averaging, which predicts V_L~5.09 km/s -- agrees with "
               "the tabulated 5.068 km/s to <0.5%. NOT independently "
               "confirmed against ballistic-phonon-imaging data the way "
               "Cu and Al are in the source paper -- treat alpha_par, "
               "alpha_perp for Sapphire/Nb as an assumption, not a fit."),
    "Ta": dict(
        V_L=4160.0, V_T=2037.0, rho=16.65e3,
        source="ADDED (not in the PCE report): Voigt-Reuss-Hill average of "
               "single-crystal Ta constants C11~267, C12~161, C44~82.5 GPa "
               "(K~196 GPa, G~69 GPa), rho=16650 kg/m^3. Approximate; "
               "verify against a Ta ultrasonic reference. Bonding alphas "
               "are an assumption (copied from the Cu fit)."),
}


# ---------------------------------------------------------------------------
# Thesis elastic constants (isotropic Lame parameters, GPa) -- Hernandez et al.,
# arXiv:2408.04732 (sapphire, Table 3) and the thesis anharmonic-downconversion table (Al, Nb, Ta).
# Densities are NOT given there: the values below are standard bulk densities (assumption).
# V_T = sqrt(mu/rho), V_L = sqrt((lambda + 2 mu)/rho).  Cu keeps the Every-Koos-Wolfe values.
# ---------------------------------------------------------------------------
import copy
MATERIALS_EKW = copy.deepcopy(MATERIALS)          # original Every-Koos-Wolfe (c-axis sapphire) set
LAME_THESIS = {                                   # (lambda GPa, mu GPa, rho kg/m^3)
    "sapphire": (139.8, 166.24, 3986.0),
    "Al": (58.7, 26.2, 2700.0),
    "Nb": (145.0, 39.7, 8570.0),
    "Ta": (149.3, 70.55, 16650.0),
}

def use_thesis_constants():
    for k, (lam, mu, rho) in LAME_THESIS.items():
        MATERIALS[k] = dict(V_L=float(np.sqrt((lam + 2 * mu) * 1e9 / rho)), V_T=float(np.sqrt(mu * 1e9 / rho)), rho=rho,
                            source="Thesis Lame parameters lambda=%.4g, mu=%.4g GPa, rho=%g kg/m^3 (isotropic)." % (lam, mu, rho))

def use_ekw_constants():
    for k, v in MATERIALS_EKW.items():
        MATERIALS[k] = copy.deepcopy(v)

use_thesis_constants()


def add_material(name, rho, lam_GPa=None, mu_GPa=None, V_L=None, V_T=None, source=""):
    """Register an isotropic material. Give either (lam_GPa, mu_GPa) or (V_L, V_T) in m/s; rho in kg/m^3."""
    if lam_GPa is not None and mu_GPa is not None:
        V_T = float(np.sqrt(mu_GPa * 1e9 / rho)); V_L = float(np.sqrt((lam_GPa + 2 * mu_GPa) * 1e9 / rho))
    elif V_L is None or V_T is None:
        raise ValueError("%s: give (lam_GPa, mu_GPa) or (V_L, V_T)" % name)
    MATERIALS[name] = dict(V_L=float(V_L), V_T=float(V_T), rho=float(rho), source=source or "user-defined")

def load_materials(path):
    """Load materials from a JSON file: {name: {"rho":..., "lambda_GPa":..., "mu_GPa":... | "V_L":..., "V_T":..., "source":...}}."""
    with open(path) as f:
        d = json.load(f)
    for name, m in d.items():
        if name.startswith("_"):
            continue
        add_material(name, m["rho"], m.get("lambda_GPa"), m.get("mu_GPa"), m.get("V_L"), m.get("V_T"), m.get("source", ""))
    return sorted(k for k in d if not k.startswith("_"))


def _lame(V_L, V_T, rho):
    mu = rho * V_T**2
    lam = rho * V_L**2 - 2 * mu
    return lam, mu


def critical_angle_deg(crystal="sapphire"):
    m = MATERIALS[crystal]
    return np.degrees(np.arcsin(m["V_T"] / m["V_L"]))


def _kz(p, V):
    """Vertical reduced wavenumber; principal branch (Im>=0) selects the
    physically correct (decaying) solution for evanescent waves."""
    return np.sqrt(complex(1.0 / V**2) - p**2 + 0j)


def _L_fields(s, p, kzL, mu, lam, V):
    ux = 1j * p
    uz = 1j * s * kzL
    sxz = -2 * mu * s * p * kzL
    szz = -(lam / V**2 + 2 * mu * kzL**2)
    return np.array([ux, uz, sxz, szz], dtype=complex)


def _T_fields(s, p, kzT, mu):
    ux = -1j * s * kzT
    uz = 1j * p
    sxz = mu * (kzT**2 - p**2)
    szz = -2 * mu * s * p * kzT
    return np.array([ux, uz, sxz, szz], dtype=complex)


def solve_interface(theta_inc_deg, incident="T", alpha_par=np.inf, alpha_perp=np.inf,
                     crystal="sapphire", metal="Cu"):
    mc = MATERIALS[crystal]
    mm = MATERIALS[metal]
    V_cL, V_cT, rho_c = mc["V_L"], mc["V_T"], mc["rho"]
    V_mL, V_mT, rho_m = mm["V_L"], mm["V_T"], mm["rho"]
    lam_c, mu_c = _lame(V_cL, V_cT, rho_c)
    lam_m, mu_m = _lame(V_mL, V_mT, rho_m)

    theta = np.radians(theta_inc_deg)
    V_inc = V_cT if incident == "T" else V_cL
    p = np.sin(theta) / V_inc

    kzL_c, kzT_c = _kz(p, V_cL), _kz(p, V_cT)
    kzL_m, kzT_m = _kz(p, V_mL), _kz(p, V_mT)

    rL = _L_fields(-1, p, kzL_c, mu_c, lam_c, V_cL)
    rT = _T_fields(-1, p, kzT_c, mu_c)
    tL = _L_fields(+1, p, kzL_m, mu_m, lam_m, V_mL)
    tT = _T_fields(+1, p, kzT_m, mu_m)

    if incident == "T":
        inc = _T_fields(+1, p, kzT_c, mu_c) * V_cT
    else:
        inc = _L_fields(+1, p, kzL_c, mu_c, lam_c, V_cL) * V_cL
    ux_i, uz_i, sxz_i, szz_i = inc

    Kpar = None if np.isinf(alpha_par) else alpha_par * mu_c / V_cT
    Kperp = None if np.isinf(alpha_perp) else alpha_perp * (lam_c + 2 * mu_c) / V_cL

    A = np.zeros((4, 4), dtype=complex)
    b = np.zeros(4, dtype=complex)

    A[0] = [rL[2], rT[2], -tL[2], -tT[2]]
    b[0] = -sxz_i
    A[1] = [rL[3], rT[3], -tL[3], -tT[3]]
    b[1] = -szz_i

    if Kpar is None:
        A[2] = [rL[0], rT[0], -tL[0], -tT[0]]
        b[2] = -ux_i
    else:
        A[2] = [rL[2] + Kpar * rL[0], rT[2] + Kpar * rT[0], -Kpar * tL[0], -Kpar * tT[0]]
        b[2] = -(sxz_i + Kpar * ux_i)

    if Kperp is None:
        A[3] = [rL[1], rT[1], -tL[1], -tT[1]]
        b[3] = -uz_i
    else:
        A[3] = [rL[3] + Kperp * rL[1], rT[3] + Kperp * rT[1], -Kperp * tL[1], -Kperp * tT[1]]
        b[3] = -(szz_i + Kperp * uz_i)

    ArL_pot, BrT_pot, AtL_pot, BtT_pot = np.linalg.solve(A, b)

    ArL, BrT, AtL, BtT = ArL_pot / V_cL, BrT_pot / V_cT, AtL_pot / V_mL, BtT_pot / V_mT

    cos_inc = np.cos(theta)

    def power(amp, rho, V, kz_val):
        cosang = V * kz_val
        if abs(cosang.imag) > 1e-9 or cosang.real < 0:
            return 0.0
        return float(np.abs(amp)**2 * (rho * V * cosang.real) / (rho_c * V_inc * cos_inc))

    W_cl = power(ArL, rho_c, V_cL, kzL_c)
    W_ct = power(BrT, rho_c, V_cT, kzT_c)
    W_ml = power(AtL, rho_m, V_mL, kzL_m)
    W_mt = power(BtT, rho_m, V_mT, kzT_m)

    return {
        "theta": theta_inc_deg, "incident": incident, "crystal": crystal, "metal": metal,
        "Gamma_cl": ArL, "Gamma_ct": BrT, "Gamma_ml": AtL, "Gamma_mt": BtT,
        "W_cl": W_cl, "W_ct": W_ct, "W_ml": W_ml, "W_mt": W_mt,
        "W_total": W_cl + W_ct + W_ml + W_mt,
        "transmission": W_ml + W_mt,
        "reflection": W_cl + W_ct,
    }


def solve_interface_SH(theta_inc_deg, alpha_SH=np.inf, crystal="sapphire", metal="Cu"):
    mc = MATERIALS[crystal]
    mm = MATERIALS[metal]
    V_cT, rho_c, mu_c = mc["V_T"], mc["rho"], mc["rho"] * mc["V_T"]**2
    V_mT, rho_m, mu_m = mm["V_T"], mm["rho"], mm["rho"] * mm["V_T"]**2

    theta = np.radians(theta_inc_deg)
    p = np.sin(theta) / V_cT
    kz_c, kz_m = _kz(p, V_cT), _kz(p, V_mT)

    A = np.zeros((2, 2), dtype=complex)
    b = np.zeros(2, dtype=complex)
    A[0] = [-1j * mu_c * kz_c, -1j * mu_m * kz_m]
    b[0] = -1j * mu_c * kz_c

    if np.isinf(alpha_SH):
        A[1] = [1, -1]
        b[1] = -1
    else:
        K_SH = alpha_SH * mu_c / V_cT
        A[1] = [-1j * mu_c * kz_c + K_SH, -K_SH]
        b[1] = -1j * mu_c * kz_c - K_SH

    R, Tr = np.linalg.solve(A, b)

    cos_inc = np.cos(theta)

    def power(amp, rho, V, kz_val):
        cosang = V * kz_val
        if abs(cosang.imag) > 1e-9 or cosang.real < 0:
            return 0.0
        return float(np.abs(amp)**2 * (rho * V * cosang.real) / (rho_c * V_cT * cos_inc))

    W_refl = power(R, rho_c, V_cT, kz_c)
    W_trans = power(Tr, rho_m, V_mT, kz_m)

    return {
        "theta": theta_inc_deg, "incident": "SH", "crystal": crystal, "metal": metal,
        "R": R, "Tr": Tr, "W_refl": W_refl, "W_trans": W_trans,
        "W_total": W_refl + W_trans, "transmission": W_trans, "reflection": W_refl,
    }


def flux_average_SH(alpha_SH, crystal="sapphire", metal="Cu", n=2000):
    thetas = np.linspace(1e-4, 90 - 1e-4, n)
    th_rad = np.radians(thetas)
    vals = np.array([solve_interface_SH(th, alpha_SH, crystal, metal)["transmission"]
                      for th in thetas])
    weight = np.cos(th_rad) * np.sin(th_rad)
    num = _trapz(vals * weight, th_rad)
    den = _trapz(weight, th_rad)
    return num / den, thetas, vals


def flux_average(incident, alpha_par, alpha_perp, crystal="sapphire", metal="Cu", n=2000):
    thetas = np.linspace(1e-4, 90 - 1e-4, n)
    th_rad = np.radians(thetas)
    vals = np.array([solve_interface(th, incident, alpha_par, alpha_perp,
                                       crystal, metal)["transmission"] for th in thetas])
    weight = np.cos(th_rad) * np.sin(th_rad)
    num = _trapz(vals * weight, th_rad)
    den = _trapz(weight, th_rad)
    return num / den, thetas, vals


def debye_weighted_average(alpha_par, alpha_perp, crystal="sapphire", metal="Cu", n=2000,
                            alpha_SH="match"):
    if alpha_SH == "match":
        alpha_SH = alpha_par
    mc = MATERIALS[crystal]
    avg_SV, _, _ = flux_average("T", alpha_par, alpha_perp, crystal, metal, n)
    avg_L, _, _ = flux_average("L", alpha_par, alpha_perp, crystal, metal, n)
    avg_SH, _, _ = flux_average_SH(alpha_SH, crystal, metal, n)
    avg_T = 0.5 * (avg_SV + avg_SH)
    w_T, w_L = 2.0 / mc["V_T"]**3, 1.0 / mc["V_L"]**3
    combined = (w_T * avg_T + w_L * avg_L) / (w_T + w_L)
    return combined, avg_SV, avg_SH, avg_L


def validate(verbose=True):
    ok = True
    thc = critical_angle_deg("sapphire")
    if verbose:
        print(f"theta_c (sapphire) = {thc:.3f} deg")
    ok &= abs(thc - np.degrees(np.arcsin(MATERIALS['sapphire']['V_T'] / MATERIALS['sapphire']['V_L']))) < 1e-9

    for theta in [0.5, 10, 20, 33, 45, 60, 80, 89]:
        for ap, aperp in [(np.inf, np.inf), (1.0, 1.1), (0.3, 0.3)]:
            r = solve_interface(theta, "T", ap, aperp)
            if abs(r["W_total"] - 1.0) > 1e-8:
                ok = False
                if verbose:
                    print(f"  ENERGY CONSERVATION FAIL at theta={theta}, "
                          f"alpha=({ap},{aperp}): W_total={r['W_total']}")
    if verbose:
        print("Energy conservation: " + ("PASS" if ok else "FAIL"))

    r = solve_interface(1e-4, "T", np.inf, np.inf)
    mc, mm = MATERIALS["sapphire"], MATERIALS["Cu"]
    Zc, Zm = mc["rho"] * mc["V_T"], mm["rho"] * mm["V_T"]
    amm = 4 * Zc * Zm / (Zc + Zm) ** 2
    match = abs(r["transmission"] - amm) < 1e-6
    ok &= match
    if verbose:
        print(f"Normal-incidence ideal-bonding T transmission: model={r['transmission']:.6f}  "
              f"textbook AMM={amm:.6f}  {'PASS' if match else 'FAIL'}")

    ok_sh = True
    for theta in [0.5, 10, 20, 33, 45, 60, 80, 89]:
        for a_sh in [np.inf, 1.0, 0.3]:
            r = solve_interface_SH(theta, a_sh)
            if abs(r["W_total"] - 1.0) > 1e-8:
                ok_sh = False
                if verbose:
                    print(f"  SH ENERGY CONSERVATION FAIL at theta={theta}, alpha_SH={a_sh}: "
                          f"W_total={r['W_total']}")
    if verbose:
        print("SH energy conservation: " + ("PASS" if ok_sh else "FAIL"))
    ok &= ok_sh

    r_sh = solve_interface_SH(1e-4, np.inf)
    match_sh = abs(r_sh["transmission"] - amm) < 1e-6
    ok &= match_sh
    if verbose:
        print(f"Normal-incidence ideal-bonding SH transmission: model={r_sh['transmission']:.6f}  "
              f"textbook AMM={amm:.6f}  {'PASS' if match_sh else 'FAIL'}")
    return ok


KAPLAN_MATERIALS = {
    "sapphire": dict(V_L=10.90e3, V_T=6.45e3, rho=3.99e3,
                      source="Kaplan, J. Low Temp. Phys. 37, 343 (1979), Table I (Al2O3)."),
    "Al": dict(V_L=6.65e3, V_T=3.26e3, rho=2.73e3, source="Kaplan (1979), Table I."),
    "Nb": dict(V_L=5.14e3, V_T=2.17e3, rho=8.59e3, source="Kaplan (1979), Table I."),
}

KAPLAN_TABLE_II_AL2O3 = {"Al": (0.50, 0.19), "Nb": (0.38, 0.091)}


def kaplan_comparison(verbose=True):
    saved = {k: MATERIALS.get(k) for k in ("sapphire", "Al", "Nb")}
    MATERIALS["sapphire"] = KAPLAN_MATERIALS["sapphire"]
    MATERIALS["Al"] = KAPLAN_MATERIALS["Al"]
    MATERIALS["Nb"] = KAPLAN_MATERIALS["Nb"]
    try:
        results = {}
        if verbose:
            print("Kaplan (1979) Table II comparison -- Al2O3 row, ideal bonding")
            print("-" * 88)
        for metal in ["Al", "Nb"]:
            eta_l, _, _ = flux_average("L", np.inf, np.inf, crystal=metal, metal="sapphire")
            eta_sv, _, _ = flux_average("T", np.inf, np.inf, crystal=metal, metal="sapphire")
            eta_sh, _, _ = flux_average_SH(np.inf, crystal=metal, metal="sapphire")
            eta_t = 0.5 * (eta_sv + eta_sh)
            k_l, k_t = KAPLAN_TABLE_II_AL2O3[metal]
            results[metal] = {"eta_l": eta_l, "eta_t": eta_t, "eta_sv": eta_sv, "eta_sh": eta_sh,
                               "kaplan_eta_l": k_l, "kaplan_eta_t": k_t}
            if verbose:
                print(f"{metal} -> Al2O3:")
                print(f"  eta_l: this solver={eta_l:.3f}   Kaplan={k_l:.3f}   "
                      f"diff={100*(eta_l-k_l)/k_l:+.1f}%")
                print(f"  eta_t: this solver={eta_t:.3f} (SV={eta_sv:.3f}, SH={eta_sh:.3f})   "
                      f"Kaplan={k_t:.3f}   diff={100*(eta_t-k_t)/k_t:+.1f}%")
        return results
    finally:
        for k, v in saved.items():
            if v is not None:
                MATERIALS[k] = v


def plot_transmission(crystal="sapphire", metal="Cu", scenarios=None, outpath=None, show=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if scenarios is None:
        scenarios = [
            (np.inf, np.inf, "Ideal / welded bonding"),
            (1.0, 1.1, r"Realistic (literature-fitted Cu): $\alpha_\parallel$=1.0, $\alpha_\perp$=1.1"),
            (1.0, 1.0, r"Generic order-unity: $\alpha_\parallel=\alpha_\perp$=1.0"),
            (0.3, 0.3, r"Very loose: $\alpha_\parallel=\alpha_\perp$=0.3"),
        ]
    thetas = np.linspace(0.1, 89.9, 400)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for ap, aperp, label in scenarios:
        vals = [solve_interface(th, "T", ap, aperp, crystal, metal)["transmission"] for th in thetas]
        ax.plot(thetas, vals, label=label)
    ax.axvline(critical_angle_deg(crystal), color="gray", linestyle=":", linewidth=1,
               label=rf"$\theta_c$ = {critical_angle_deg(crystal):.1f} deg")
    ax.set_xlabel(r"Incidence angle $\theta$ (deg), from the interface normal")
    ax.set_ylabel(r"T-wave transmission probability into the metal, $\phi_{ct}(\theta)$")
    ax.set_title(f"{crystal.capitalize()} / {metal}: critical-cone channeling")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
        print(f"Saved {outpath}")
    if show:
        plt.show()
    return fig


def print_summary(crystal="sapphire", metal="Cu", scenarios=None):
    if scenarios is None:
        scenarios = [
            (np.inf, np.inf, "Ideal / welded bonding"),
            (1.0, 1.1, "Realistic (literature-fitted Cu): alpha_par=1.0, alpha_perp=1.1"),
            (1.0, 1.0, "Generic order-unity: alpha_par=alpha_perp=1.0"),
            (0.3, 0.3, "Very loose: alpha_par=alpha_perp=0.3"),
        ]
    print(f"\n{crystal.capitalize()} / {metal}  "
          f"(V_L={MATERIALS[metal]['V_L']/1e3:.3f} km/s, V_T={MATERIALS[metal]['V_T']/1e3:.3f} km/s, "
          f"rho={MATERIALS[metal]['rho']/1e3:.3f} g/cm^3)")
    print(f"  source: {MATERIALS[metal]['source']}")
    for ap, aperp, label in scenarios:
        combined, avg_SV, avg_SH, avg_L = debye_weighted_average(ap, aperp, crystal, metal)
        print(f"  {label}")
        print(f"      <phi>_SV-incident={avg_SV:.4f}  <phi>_SH-incident={avg_SH:.4f}  "
              f"<phi>_L-incident={avg_L:.4f}  Debye-weighted(1L+1SV+1SH)={combined:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Every-Koos-Wolfe (1984) critical-cone-channeling / loose-bonding phonon "
                    "interface-transmission model.")
    parser.add_argument("--crystal", default="sapphire", choices=list(MATERIALS))
    parser.add_argument("--metal", default=None, choices=list(MATERIALS),
                        help="Default: run Cu, Al, Nb, Ta.")
    parser.add_argument("--alpha-par", type=float, default=None)
    parser.add_argument("--alpha-perp", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--kaplan-check", action="store_true")
    args = parser.parse_args()

    print("Step 1: validating the solver")
    ok = validate()
    if not ok:
        raise SystemExit("Validation FAILED -- see output above.")

    if args.kaplan_check:
        kaplan_comparison()
        raise SystemExit(0)

    scenarios = None
    if args.alpha_par is not None:
        aperp = args.alpha_perp if args.alpha_perp is not None else args.alpha_par
        scenarios = [(args.alpha_par, aperp, f"alpha_par={args.alpha_par}, alpha_perp={aperp}")]

    metals = [args.metal] if args.metal else ["Cu", "Al", "Nb", "Ta"]
    print("Step 2: flux-averaged transmission coefficients")
    for metal in metals:
        print_summary(args.crystal, metal, scenarios)

    if not args.no_plot:
        import os
        for metal in metals:
            outpath = os.path.join(args.outdir, f"transmission_{args.crystal}_{metal}.png")
            plot_transmission(args.crystal, metal, scenarios, outpath=outpath, show=False)

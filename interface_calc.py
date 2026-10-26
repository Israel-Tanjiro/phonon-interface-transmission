#!/usr/bin/env python3
"""interface_calc.py -- phonon transmission across a loosely bonded crystal/metal interface.

Loose-bond acoustic mismatch model (Every, Koos & Wolfe, PRB 29, 2190, 1984) with L, SV and SH channels,
flux averaging (cos(theta) sin(theta) weight) and Debye weighting (1 L + 2 T).

Examples
--------
  python3 interface_calc.py --validate
  python3 interface_calc.py --crystal sapphire --metal Ta
  python3 interface_calc.py --crystal sapphire --metal Ta Al Nb --alpha-par 1.0 --alpha-perp 1.1
  python3 interface_calc.py --materials materials/example_new_pair.json --crystal silicon_iso --metal Au --figure

Output (in --outdir): summary.csv with the averaged probabilities, and, with --figure,
transmission_<crystal>_<metal>.csv/.png with the angle-resolved transmission.
"""
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phonon_interface_model as pim

DEFAULT_SCENARIOS = [
    (np.inf, np.inf, "welded"),
    (1.0, 1.1, "alpha_par=1.0, alpha_perp=1.1"),
    (1.0, 1.0, "alpha_par=alpha_perp=1.0"),
    (0.3, 0.3, "alpha_par=alpha_perp=0.3"),
]

def scenarios_from_args(a):
    if a.alpha_par is None:
        return DEFAULT_SCENARIOS
    ap = a.alpha_par; aperp = a.alpha_perp if a.alpha_perp is not None else ap
    return [(np.inf, np.inf, "welded"), (ap, aperp, "alpha_par=%g, alpha_perp=%g" % (ap, aperp))]

def material_report(name):
    m = pim.MATERIALS[name]
    lam, mu = m["rho"] * m["V_L"] ** 2 - 2 * m["rho"] * m["V_T"] ** 2, m["rho"] * m["V_T"] ** 2
    return ("%-9s rho=%8.1f kg/m^3  V_L=%7.1f  V_T=%7.1f m/s  (lambda=%6.1f, mu=%6.1f GPa)"
            % (name, m["rho"], m["V_L"], m["V_T"], lam / 1e9, mu / 1e9))

def run_pair(crystal, metal, scen, n, alpha_sh, outdir, figure):
    rows = []
    print("\n%s -> %s" % (crystal, metal))
    print("  " + material_report(crystal)); print("  " + material_report(metal))
    print("  critical angle for incident T wave: %.2f deg" % pim.critical_angle_deg(crystal))
    print("  %-30s %8s %8s %8s %10s" % ("scenario", "<SV>", "<SH>", "<L>", "Debye"))
    for ap, aperp, label in scen:
        comb, sv, sh, l = pim.debye_weighted_average(ap, aperp, crystal, metal, n=n, alpha_SH=ap if alpha_sh == "match" else alpha_sh)
        print("  %-30s %8.4f %8.4f %8.4f %10.4f" % (label, sv, sh, l, comb))
        rows.append(dict(crystal=crystal, metal=metal, scenario=label, avg_SV=sv, avg_SH=sh, avg_L=l, debye=comb))
    if figure:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        th = np.linspace(0.2, 89.8, n)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        out = {"theta_deg": th}
        for k, (inc, title) in enumerate((("T", "SV wave incident"), ("L", "L wave incident"))):
            for ap, aperp, label in scen:
                tr = np.array([pim.solve_interface(t, inc, ap, aperp, crystal, metal)["transmission"] for t in th])
                ax[k].plot(th, tr, lw=1.8, label=label); out["%s_%s" % (inc, label)] = tr
            ax[k].axvline(pim.critical_angle_deg(crystal), c="gray", ls=":")
            ax[k].set_xlabel(r"incidence angle $\theta_i$ (deg)"); ax[k].set_title("%s -> %s, %s" % (crystal, metal, title), fontsize=10)
            ax[k].set_xlim(0, 90); ax[k].set_ylim(0, 1.02)
        ax[0].set_ylabel("transmitted power"); ax[0].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        base = os.path.join(outdir, "transmission_%s_%s" % (crystal, metal))
        fig.savefig(base + ".png", dpi=200); plt.close(fig)
        keys = list(out)
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(keys)
            for i in range(len(th)): w.writerow(["%.6g" % out[k][i] for k in keys])
        print("  wrote %s.png and .csv" % base)
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", default=os.path.join(HERE, "materials", "thesis_constants.json"),
                    help="JSON file with materials (default: materials/thesis_constants.json)")
    ap.add_argument("--crystal", default="sapphire"); ap.add_argument("--metal", nargs="+", default=None)
    ap.add_argument("--alpha-par", type=float, default=None); ap.add_argument("--alpha-perp", type=float, default=None)
    ap.add_argument("--alpha-sh", default="match", help="'match' (= alpha_par, default) or a number")
    ap.add_argument("--n", type=int, default=800, help="points in the angular grid")
    ap.add_argument("--outdir", default="output"); ap.add_argument("--figure", action="store_true")
    ap.add_argument("--validate", action="store_true", help="run the built-in checks and exit")
    ap.add_argument("--list", action="store_true", help="list the materials in the file and exit")
    a = ap.parse_args()

    names = pim.load_materials(a.materials)
    if a.list:
        for k in names: print(material_report(k))
        return
    if a.validate:
        ok = pim.validate(); sys.exit(0 if ok else 1)
    if a.crystal not in pim.MATERIALS: sys.exit("unknown crystal '%s'; available: %s" % (a.crystal, ", ".join(names)))
    metals = a.metal or [k for k in names if k != a.crystal]
    for m in metals:
        if m not in pim.MATERIALS: sys.exit("unknown metal '%s'; available: %s" % (m, ", ".join(names)))
    alpha_sh = a.alpha_sh if a.alpha_sh == "match" else float(a.alpha_sh)
    os.makedirs(a.outdir, exist_ok=True)
    allrows = []
    for m in metals:
        allrows += run_pair(a.crystal, m, scenarios_from_args(a), a.n, alpha_sh, a.outdir, a.figure)
    with open(os.path.join(a.outdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print("\nwrote %s" % os.path.join(a.outdir, "summary.csv"))

if __name__ == "__main__":
    main()

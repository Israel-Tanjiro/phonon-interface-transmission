"""Independent checks of the interface solver against closed-form results.

Run with:  python3 tests/test_interface.py      (or: python3 -m pytest tests)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phonon_interface_model as pim

pim.load_materials(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "materials", "thesis_constants.json"))
CR, ME = "sapphire", "Ta"

def test_energy_conservation():
    for th in (1, 10, 25, 36.3, 36.5, 50, 80, 89):
        for inc in ("T", "L"):
            for ap in ((np.inf, np.inf), (1.0, 1.1), (0.3, 0.3), (0.0, 0.0)):
                r = pim.solve_interface(th, inc, ap[0], ap[1], CR, ME)
                assert abs(r["W_total"] - 1) < 1e-8, (th, inc, ap, r["W_total"])
        for a in (np.inf, 1.0, 0.3):
            assert abs(pim.solve_interface_SH(th, a, CR, ME)["W_total"] - 1) < 1e-8

def test_normal_incidence_closed_form():
    c, m = pim.MATERIALS[CR], pim.MATERIALS[ME]
    r = m["rho"] * m["V_T"] / (c["rho"] * c["V_T"])
    for a in (0.3, 1.0, 3.0, np.inf):
        expect = 4 * r / ((1 + r) ** 2 + (0 if np.isinf(a) else r * r / a ** 2))
        assert abs(pim.solve_interface(1e-4, "T", a, a, CR, ME)["transmission"] - expect) < 1e-6
        assert abs(pim.solve_interface_SH(1e-4, a, CR, ME)["transmission"] - expect) < 1e-6

def test_SH_closed_form():
    c, m = pim.MATERIALS[CR], pim.MATERIALS[ME]
    for th in (10, 40, 70):
        for a in (0.3, 1.0):
            p = np.sin(np.radians(th)) / c["V_T"]
            kc, km = np.sqrt(1 / c["V_T"] ** 2 - p ** 2 + 0j), np.sqrt(1 / m["V_T"] ** 2 - p ** 2 + 0j)
            A, B, K = c["rho"] * c["V_T"] ** 2 * kc, m["rho"] * m["V_T"] ** 2 * km, a * c["rho"] * c["V_T"]
            Tr = 2 * K * A / (K * (A + B) - 1j * A * B)
            W = abs(Tr) ** 2 * m["rho"] * m["V_T"] * (m["V_T"] * km).real / (c["rho"] * c["V_T"] * np.cos(np.radians(th)))
            assert abs(W - pim.solve_interface_SH(th, a, CR, ME)["transmission"]) < 1e-9

def test_free_surface_closed_form():
    c = pim.MATERIALS[CR]; k = c["V_L"] / c["V_T"]
    for th in (10, 25, 40, 60):
        t = np.radians(th); sL = k * np.sin(t); s2L = 2 * sL * np.sqrt(1 - sL ** 2 + 0j)
        den = np.sin(2 * t) * s2L + k ** 2 * np.cos(2 * t) ** 2
        r = pim.solve_interface(th, "T", 0.0, 0.0, CR, ME)
        assert abs(abs(k * np.sin(4 * t) / den) - abs(r["Gamma_cl"])) < 1e-9
        assert abs((np.sin(2 * t) * s2L - k ** 2 * np.cos(2 * t) ** 2) / den - r["Gamma_ct"]) < 1e-9

def test_welded_limit_is_large_alpha():
    a = pim.solve_interface(30, "T", np.inf, np.inf, CR, ME)["transmission"]
    b = pim.solve_interface(30, "T", 1e6, 1e6, CR, ME)["transmission"]
    assert abs(a - b) < 1e-5

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)

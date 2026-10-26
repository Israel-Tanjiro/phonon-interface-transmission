# phonon-interface-transmission

Transmission probability of acoustic phonons across a planar interface between a crystal (substrate) and a
metal film, using the **loose-bond acoustic mismatch model** of Every, Koos and Wolfe
(Phys. Rev. B **29**, 2190, 1984): two isotropic half-spaces joined by distributed springs with tangential
and normal stiffness. L, SV and SH channels are solved exactly (complex 4x4 and 2x2 linear systems),
then averaged over angle (flux weight `cos(theta) sin(theta)`) and over modes (Debye weights, 1 L + 2 T).

It was written to supply the interface absorption probability `P_a` used in quasiparticle-collection
simulations of superconducting qubits (sapphire / Ta, Al, Nb films), and is meant to be reusable for any
other pair of isotropic materials.

## Quick start

```bash
pip install -r requirements.txt            # numpy, matplotlib
python3 interface_calc.py --validate       # run the built-in checks
python3 interface_calc.py --crystal sapphire --metal Ta Al Nb
python3 interface_calc.py --crystal sapphire --metal Ta --alpha-par 1.0 --alpha-perp 1.1 --figure
python3 tests/test_interface.py            # independent closed-form tests
python3 phonon_interface_figures.py --outdir figures   # the thesis figures
```

Example output (thesis constants):

```
sapphire -> Ta
  critical angle for incident T wave: 36.39 deg
  scenario                           <SV>     <SH>      <L>      Debye
  welded                           0.8502   0.8525   0.8955     0.8555
  alpha_par=1.0, alpha_perp=1.1    0.8236   0.7072   0.7100     0.7602
  alpha_par=alpha_perp=1.0         0.8145   0.7072   0.6887     0.7540
  alpha_par=alpha_perp=0.3         0.2848   0.2815   0.2225     0.2774
```

`Debye` is the number to use as the transmission (absorption) probability; `<SV>`, `<SH>`, `<L>` are the
flux-averaged probabilities for each incident mode.

## Inputs

Each material needs a density `rho` (kg/m^3) and **either** the Lame parameters `lambda_GPa`, `mu_GPa`
**or** the sound speeds `V_L`, `V_T` (m/s), in a JSON file (see `materials/`):

```json
{
  "sapphire": {"rho": 3986, "lambda_GPa": 139.8, "mu_GPa": 166.24, "source": "..."},
  "Ta":       {"rho": 16650, "lambda_GPa": 149.3, "mu_GPa": 70.55,  "source": "..."}
}
```

Speeds follow from `V_T = sqrt(mu/rho)` and `V_L = sqrt((lambda + 2 mu)/rho)`.
The coupling of the interface is set by two dimensionless numbers,

```
alpha_par  = K_par  / (rho_c V_T,c w),      alpha_perp = K_perp / (rho_c V_L,c w)
```

(`K` = spring stiffness per unit area, `w` = phonon angular frequency, `c` = crystal). `alpha -> infinity`
is a welded interface (classical acoustic mismatch), `alpha -> 0` a free surface. **The elastic constants do
not determine alpha**; it must be measured, fitted or scanned (see Limitations).

## Using it for another interface

1. Copy `materials/example_new_pair.json`, rename the materials, and fill in `rho` and either
   `lambda_GPa`+`mu_GPa` or `V_L`+`V_T`. Cite the source of each number in `"source"`.
2. Run
   `python3 interface_calc.py --materials my_materials.json --crystal <substrate> --metal <film> --figure`.
3. Read the `Debye` column for the transmission probability and `output/transmission_*.png` for the angular
   dependence. Scan `--alpha-par/--alpha-perp` to see the sensitivity; the model has no other free parameter
   (SH uses `alpha_par` by default, `--alpha-sh` overrides).
4. Run `python3 interface_calc.py --validate` and `python3 tests/test_interface.py` after changing any code.
5. From Python:

```python
import phonon_interface_model as pim
pim.load_materials("my_materials.json")           # or pim.add_material(name, rho, lam_GPa=..., mu_GPa=...)
W, sv, sh, L = pim.debye_weighted_average(1.0, 1.1, crystal="sapphire", metal="Ta")
r = pim.solve_interface(30.0, "T", 1.0, 1.1, "sapphire", "Ta")   # one angle; r["transmission"], r["W_*"], r["Gamma_*"]
```

The first argument pair of `debye_weighted_average` is `(alpha_par, alpha_perp)`; use `numpy.inf` for welded.

## What the code computes

For an incident L or SV wave (angle `theta` from the normal) in the crystal it solves for the four outgoing
waves (reflected L, SV; transmitted L, SV) from continuity of traction and the spring law
`sigma_xz = K_par du_x`, `sigma_zz = K_perp du_z`, and returns amplitudes `Gamma_*` (relative displacement)
and power fractions `W_* = |Gamma|^2 rho_j V_j cos(theta_j) / (rho_i V_i cos(theta_i))` (zero for evanescent
waves). `W_cl + W_ct + W_ml + W_mt = 1`. See `docs/technical_note.tex` (derivation, 8 pages) and
`docs/thesis_appendix.tex` (compact version).

## Validation

`--validate` and `tests/test_interface.py` check: energy conservation to 1e-8 (welded, spring, free surface,
SH); the normal-incidence closed form `4r/((1+r)^2 + r^2/alpha^2)`; the SH closed form; the free-surface
amplitudes of an incident SV wave (Rayleigh-type determinant); and the welded limit. Kaplan's
(J. Low Temp. Phys. 37, 343, 1979) welded-interface values for Al/Nb on sapphire are reproduced to within
about 1.3% (`python3 phonon_interface_model.py --kaplan-check`).

## Limitations (read before quoting numbers)

* Both media are **isotropic**. Sapphire is anisotropic; the Lame parameters in `thesis_constants.json` are
  an averaged set, and `ekw1984.json` holds the c-axis values of Every et al. The choice changes the
  critical angle (36.4 vs 33 deg) and moves the transmission by a few percent.
* The interface is planar and infinitely thin; film thickness, roughness, interlayers (oxide) and phonon
  focusing are not modelled.
* `alpha` is an input. Only Cu has literature fits (Every et al.); values for other films are scanned.
  If `K` is frequency independent then `alpha ~ 1/w`, so the transmission depends on phonon energy.
* SH waves use `alpha_SH = alpha_par` unless `--alpha-sh` is given (assumption).
* Densities in `thesis_constants.json` are standard bulk values, not taken from the source tables.
* Ta, Al, Nb parameters are those of the thesis calculation; Cu is from Every et al.

## Layout

```
interface_calc.py              command-line tool (tables, CSV, plots)
phonon_interface_model.py      solver: fields, 4x4/2x2 systems, averages, validation, Kaplan check
phonon_interface_figures.py    figures of the thesis chapter
materials/                     thesis_constants.json, ekw1984.json, example_new_pair.json
tests/test_interface.py        independent closed-form tests
docs/technical_note.tex/.pdf   full derivation          docs/thesis_appendix.tex   compact appendix
docs/thesis_main_section.tex   main-text section as used in the thesis
figures/                       figures produced by phonon_interface_figures.py
```

## Citing

See `CITATION.cff`. For a permanent reference, create a GitHub release and archive it with Zenodo
(Settings -> Integrations -> Zenodo), then cite the DOI it gives you together with Every, Koos and Wolfe
(PRB 29, 2190, 1984).

## References

* A. G. Every, G. L. Koos, J. P. Wolfe, Phys. Rev. B 29, 2190 (1984).
* S. B. Kaplan, J. Low Temp. Phys. 37, 343 (1979).
* B. A. Auld, *Acoustic Fields and Waves in Solids*, Vol. II (Wiley, 1973).
* I. Hernandez et al., arXiv:2408.04732 (sapphire elastic constants).
# phonon-interface-transmission

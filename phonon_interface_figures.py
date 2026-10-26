#!/usr/bin/env python3
"""Figures for the sapphire/metal phonon-interface section (Every-Koos-Wolfe loose-bonding model).

Fig. 1  interface_concept.png/pdf   (a) reflection and refraction of an L wave at a sapphire/Ta interface, with the
                                    spring (loose-bond) boundary condition; (b) slowness construction: conserved k_par,
                                    critical angle theta_c for an incident transverse wave
Fig. 2  interface_angular.png/pdf   (a,b) energy partition of an incident L wave, welded and loosely bonded;
                                    (c) reflected-L amplitude for an incident SV wave (pseudo-surface-wave resonance);
                                    (d) SV transmission versus angle for several bonding strengths
Fig. 3  interface_vs_alpha.png/pdf  flux- and Debye-averaged transmission versus coupling alpha (Cu, Al, Ta)

usage : python3 phonon_interface_figures.py [--metal Ta] [--outdir figures] [--n 600] [--materials materials/thesis_constants.json]
needs : numpy, matplotlib, and phonon_interface_model.py (the solver) in the same folder
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle
import phonon_interface_model as pim

C = {'L': '#0072B2', 'SV': '#D55E00', 'T': '#009E73', 'gray': '#555555'}

def zigzag(ax, x0, y0, x1, y1, n=6, amp=0.025, **kw):
    t = np.linspace(0, 1, 2 * n + 1)
    dx, dy = x1 - x0, y1 - y0; L = np.hypot(dx, dy); nx, ny = -dy / L, dx / L
    off = amp * np.where(np.arange(2 * n + 1) % 2 == 1, 1, -1) * (np.arange(2 * n + 1) > 0) * (np.arange(2 * n + 1) < 2 * n)
    ax.plot(x0 + t * dx + off * nx, y0 + t * dy + off * ny, **kw)

def draw_ray_panel(a, metal, theta_i_deg=50.0, title=True):
    """Real-space ray diagram with the loose-bond (spring) inset; draws into axes a."""
    mc, mm = pim.MATERIALS['sapphire'], pim.MATERIALS[metal]
    a.set_xlim(-1.15, 1.15); a.set_ylim(-0.95, 1.08); a.set_aspect('equal'); a.axis('off')
    a.add_patch(Rectangle((-1.15, 0.03), 2.3, 1.05, fc='#e8f1fa', ec='none', zorder=0))
    a.add_patch(Rectangle((-1.15, -0.95), 2.3, 0.92, fc='#f3ead6', ec='none', zorder=0))
    a.plot([-1.15, 1.15], [0.03, 0.03], c='k', lw=1.2); a.plot([-1.15, 1.15], [-0.03, -0.03], c='k', lw=1.2)
    for x in np.arange(-1.12, 1.15, 0.07): a.plot([x, x - 0.04], [-0.03, -0.09], c='k', lw=0.7)
    a.text(-1.1, 1.03, r'sapphire  ($\rho_c,\ \lambda_c,\ \mu_c$)', fontsize=10, va='top')
    a.text(1.1, -0.9, r'%s  ($\rho_m,\ \lambda_m,\ \mu_m$)' % metal, fontsize=10, va='bottom', ha='right')
    a.add_patch(FancyArrowPatch((0, 0.03), (0, 1.0), arrowstyle='-|>', mutation_scale=12, color='k', lw=1.3, zorder=5)); a.text(0.03, 0.97, r'$\hat n$', fontsize=11)
    a.plot([0, 0], [-0.03, -0.9], c='k', lw=0.8, ls=':')
    th = np.radians(theta_i_deg); vL, vT = mc['V_L'], mc['V_T']; L = 0.80
    def ray(v_j, up, col, lab, lab_off=(0, 0), lw=2.0):
        tj = np.arcsin(np.sin(th) * v_j / vL)
        y0 = 0.03 if up else -0.03
        dx, dy = np.sin(tj), (1 if up else -1) * np.cos(tj)
        a.add_patch(FancyArrowPatch((0, y0), (L * dx, y0 + L * dy), arrowstyle='-|>', mutation_scale=13, color=col, lw=lw, shrinkA=0, shrinkB=0, zorder=4))
        a.text(L * dx + lab_off[0], y0 + L * dy + lab_off[1], lab, color=col, fontsize=11, ha='center', va='center', weight='bold')
        return tj
    dx, dy = -np.sin(th), np.cos(th)
    a.add_patch(FancyArrowPatch((L * dx, 0.03 + L * dy), (0, 0.03), arrowstyle='-|>', mutation_scale=13, color=C['L'], lw=2.4, shrinkA=0, shrinkB=0, zorder=4))
    a.text(L * dx - 0.02, 0.03 + L * dy + 0.07, 'incident L', color=C['L'], fontsize=11, ha='center', weight='bold')
    t_cl = ray(vL, True, C['L'], 'L', (0.06, 0.02)); t_ct = ray(vT, True, C['SV'], 'SV', (0.04, 0.06))
    t_ml = ray(mm['V_L'], False, C['L'], 'L', (0.07, -0.02)); t_mt = ray(mm['V_T'], False, C['SV'], 'SV', (-0.07, -0.03))
    def arc(t, side, up, r, col, lab, lab_xy):
        y0 = 0.03 if up else -0.03; d = np.degrees(t)
        lo, hi = ((90 - d, 90) if (up and side > 0) else (90, 90 + d) if up else (-90, -90 + d))
        a.add_patch(Arc((0, y0), 2 * r, 2 * r, theta1=lo, theta2=hi, color=col, lw=1.1, zorder=3))
        a.text(lab_xy[0], lab_xy[1], lab, fontsize=9.5, color=col, ha='center', va='center')
    arc(th, -1, True, 0.30, C['L'], r'$\theta_i$', (-0.20, 0.40)); arc(t_cl, +1, True, 0.30, C['L'], r'$\theta_{cL}$', (0.25, 0.40))
    arc(t_ct, +1, True, 0.45, C['SV'], r'$\theta_{cT}$', (0.17, 0.60))
    arc(t_ml, +1, False, 0.30, C['L'], r'$\theta_{mL}$', (0.20, -0.42)); arc(t_mt, +1, False, 0.50, C['SV'], r'$\theta_{mT}$', (-0.10, -0.55))
    a.text(1.1, -0.10, r'Snell: $\sin\theta_j/v_j=\sin\theta_i/v_i$', fontsize=10, ha='right', va='top')
    ins = a.inset_axes([0.02, 0.02, 0.38, 0.30]); ins.set_xlim(0, 1); ins.set_ylim(-0.1, 1.15); ins.axis('off')
    ins.add_patch(Rectangle((0.0, 0.66), 1.0, 0.34, fc='#e8f1fa', ec='k', lw=0.8)); ins.add_patch(Rectangle((0.0, 0.0), 1.0, 0.34, fc='#fbf6ea', ec='k', lw=0.8))
    for x in (0.15, 0.45): zigzag(ins, x, 0.34, x, 0.66, n=4, amp=0.035, c='k', lw=1.0)
    ins.text(0.30, 0.50, r'$K_\perp$', fontsize=9, ha='center', va='center')
    ins.annotate('', xy=(0.95, 0.50), xytext=(0.62, 0.50), arrowprops=dict(arrowstyle='<->', lw=1.1)); ins.text(0.785, 0.62, r'$K_\parallel$', fontsize=9, ha='center', va='bottom')
    ins.text(0.5, 1.03, 'loose bond: distributed springs', fontsize=8, ha='center', va='bottom')
    ins.text(0.5, -0.04, r'$\sigma_{zz}=K_\perp\Delta u_z,\ \sigma_{xz}=K_\parallel\Delta u_x$', fontsize=8, ha='center', va='top')
    if title: a.set_title(r'(a) Incident L wave at the sapphire/%s interface' % metal, fontsize=11, y=1.01)

def fig_concept(metal, outdir, theta_i_deg=50.0):
    mc, mm = pim.MATERIALS['sapphire'], pim.MATERIALS[metal]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.15, 1]})
    draw_ray_panel(ax[0], metal, theta_i_deg); th = np.radians(theta_i_deg)
    # ---------------- (b) slowness construction (s/km)
    b = ax[1]; b.set_aspect('equal')
    s_cL, s_cT, s_mL, s_mT = [1e3 / v for v in (mc['V_L'], mc['V_T'], mm['V_L'], mm['V_T'])]
    up = np.linspace(0, np.pi, 400)
    b.set_xlim(-0.30, 0.30); b.set_ylim(-0.40, 0.22)
    b.plot(s_cL * np.cos(up), s_cL * np.sin(up), c=C['L'], lw=2.2); b.plot(s_cT * np.cos(up), s_cT * np.sin(up), c=C['SV'], lw=2.2)
    b.plot(s_mL * np.cos(up), -s_mL * np.sin(up), c=C['L'], lw=2.2, ls='--'); b.plot(s_mT * np.cos(up), -s_mT * np.sin(up), c=C['SV'], lw=2.2, ls='--')
    b.axhline(0, c='k', lw=1.2)
    b.text(0.0, s_cL + 0.006, 'sapphire L', color=C['L'], fontsize=8.5, ha='center'); b.text(0.0, s_cT + 0.006, 'sapphire SV', color=C['SV'], fontsize=8.5, ha='center')
    b.text(0.0, -s_mL - 0.006, '%s L' % metal, color=C['L'], fontsize=8.5, ha='center', va='top'); b.text(-0.22, -0.375, '%s SV (radius %.2f, partly off scale)' % (metal, s_mT), color=C['SV'], fontsize=8, ha='left')
    p1 = np.sin(th) * s_cL
    b.plot([p1, p1], [-np.sqrt(s_mT ** 2 - p1 ** 2), np.sqrt(s_cT ** 2 - p1 ** 2)], c='k', lw=1.0, ls=':')
    for r_, sg, col in ((s_cL, 1, C['L']), (s_cT, 1, C['SV']), (s_mL, -1, C['L']), (s_mT, -1, C['SV'])):
        b.plot([p1], [sg * np.sqrt(r_ ** 2 - p1 ** 2)], 'o', c=col, ms=5, zorder=5)
    b.text(p1 + 0.006, -0.065, r'$p=\sin\theta_i/v_{cL}$' + '\n(incident L, $%d^\\circ$):\nall four waves propagate' % round(theta_i_deg), fontsize=8, ha='left', va='top')
    p2 = s_cL; thc = np.degrees(np.arcsin(mc['V_T'] / mc['V_L']))
    b.plot([0, p2], [0, np.sqrt(s_cT ** 2 - p2 ** 2)], c=C['SV'], lw=1.8)
    b.add_patch(Arc((0, 0), 0.07, 0.07, theta1=90 - thc, theta2=90, color=C['SV'], lw=1.2)); b.text(0.006, 0.043, r'$\theta_c$', color=C['SV'], fontsize=10)
    b.plot([p2, p2], [0, np.sqrt(s_cT ** 2 - p2 ** 2)], c=C['SV'], lw=0.9, ls=':')
    b.text(0.115, 0.115, r'incident SV at $\theta_c=%d^\circ$:' % round(thc) + '\n$p=1/v_{cL}$ (L grazing).\nFor $\\theta>\\theta_c$ the reflected L\nwave is evanescent', fontsize=8, color=C['SV'], ha='left', va='bottom')
    b.set_xlabel(r'$p=\sin\theta/v$ = conserved $k_\parallel/\omega$  (s/km)'); b.set_yticks([]); b.set_ylabel(r'$\pm k_\perp/\omega$')
    b.set_title(r'(b) Slowness construction', fontsize=11)
    fig.tight_layout()
    for e in ('png', 'pdf'): fig.savefig(os.path.join(outdir, 'interface_concept.' + e), dpi=200)
    plt.close(fig)

def fig_angular(metal, outdir, n):
    th = np.linspace(0.2, 89.8, n); thc = pim.critical_angle_deg('sapphire')
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.10, wspace=0.28)
    # (a) large ray diagram, centred on the top row
    a = fig.add_subplot(gs[0, :]); draw_ray_panel(a, metal, 50.0, title=False)
    a.set_title(r'(a) Incident L wave at the sapphire/%s interface' % metal, fontsize=13)
    # (c) SV resonance, with (b) as an inset
    a = fig.add_subplot(gs[1, 0])
    for (ap, aperp, lab, col) in ((0.0, 0.0, 'free surface', '#000000'), (np.inf, np.inf, 'welded', C['T']),
                                  (1.0, 1.1, r'$\alpha_\parallel=1.0,\ \alpha_\perp=1.1$', C['SV']), (0.3, 0.3, r'$\alpha_\parallel=\alpha_\perp=0.3$', C['L'])):
        g = np.array([abs(pim.solve_interface(t, 'T', ap, aperp, 'sapphire', metal)['Gamma_cl']) ** 2 for t in th])
        a.plot(th, g, c=col, lw=1.8, label=lab)
    a.axvline(thc, c='gray', ls=':'); a.text(thc + 0.8, a.get_ylim()[1] * 0.9, r'$\theta_c=%.0f^\circ$' % thc, fontsize=10, color='gray')
    a.set_xlabel(r'incidence angle $\theta_i$ (deg)'); a.set_ylabel(r'$|\Gamma_{T\to L}|^2$ (reflected L)')
    a.set_title(r'(c) SV incident: pseudo-surface-wave resonance at $\theta_c$', fontsize=12); a.set_xlim(0, 90)
    a.legend(fontsize=9, loc='upper left', frameon=False)
    ins = a.inset_axes([0.64, 0.33, 0.34, 0.50])
    cols = {'cl': '#56B4E9', 'ct': '#E69F00', 'ml': '#0072B2', 'mt': '#D55E00'}
    labs = {'cl': 'refl. L', 'ct': 'refl. SV', 'ml': 'trans. L', 'mt': 'trans. SV'}
    W = {k: np.array([pim.solve_interface(t, 'L', 1.0, 1.1, 'sapphire', metal)['W_' + k] for t in th]) for k in cols}
    ins.stackplot(th, [W['ml'], W['mt'], W['cl'], W['ct']], colors=[cols['ml'], cols['mt'], cols['cl'], cols['ct']],
                  labels=[labs['ml'], labs['mt'], labs['cl'], labs['ct']], alpha=0.9, lw=0)
    ins.set_ylim(0, 1); ins.set_xlim(0, 90); ins.tick_params(labelsize=10)
    ins.set_xlabel(r'$\theta_i$ (deg)', fontsize=11); ins.set_ylabel('fraction of power', fontsize=11)
    ins.set_title('(b) L incident\n' + r'$\alpha_\parallel=1.0,\ \alpha_\perp=1.1$', fontsize=9)
    ins.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ins.patch.set_zorder(10)
    # (d) SV transmission
    a = fig.add_subplot(gs[1, 1])
    for (ap, aperp, lab, col) in ((np.inf, np.inf, 'welded', C['T']), (1.0, 1.1, r'$\alpha_\parallel=1.0,\ \alpha_\perp=1.1$', C['SV']), (0.3, 0.3, r'$\alpha_\parallel=\alpha_\perp=0.3$', C['L'])):
        tr = np.array([pim.solve_interface(t, 'T', ap, aperp, 'sapphire', metal)['transmission'] for t in th])
        a.plot(th, tr, c=col, lw=1.8, label=lab)
    a.axvline(thc, c='gray', ls=':'); a.set_ylim(0, 1.02); a.set_xlim(0, 90)
    a.set_xlabel(r'incidence angle $\theta_i$ (deg)'); a.set_ylabel(r'transmitted power into %s' % metal)
    a.set_title(r'(d) SV incident: transmission versus angle', fontsize=12); a.legend(fontsize=9, loc='lower left')
    fig.suptitle('Sapphire / %s, isotropic elastic constants' % metal, fontsize=13, y=0.93)
    for e in ('png', 'pdf'): fig.savefig(os.path.join(outdir, 'interface_angular.' + e), dpi=200, bbox_inches='tight')
    plt.close(fig)

def fig_vs_alpha(outdir, n, e0_meV=2.48):
    alphas = np.logspace(-1.5, 1.3, 36)
    fig, a = plt.subplots(figsize=(9, 6))
    col = {'Cu': '#CC79A7', 'Al': '#009E73', 'Ta': '#D55E00'}
    for metal in ('Cu', 'Al', 'Ta'):
        y = [pim.debye_weighted_average(al, al, 'sapphire', metal, n=n)[0] for al in alphas]
        a.plot(alphas, y, c=col[metal], lw=2, label='sapphire / %s' % metal)
        w = pim.debye_weighted_average(np.inf, np.inf, 'sapphire', metal, n=n)[0]
        a.axhline(w, c=col[metal], lw=0.8, ls=':')
    a.set_xscale('log'); a.set_ylim(0, 1); a.set_xlim(alphas[0], alphas[-1])
    a.set_xlabel(r'coupling  $\alpha=\alpha_\parallel=\alpha_\perp=K/(\rho v\,\omega)$'); a.set_ylabel(r'$\langle\phi\rangle_{\rm Debye}$')
    a.axvline(1.0, c='gray', ls='--', lw=0.8); a.axvline(0.3, c='gray', ls='--', lw=0.8)
    a.text(1.0, 0.04, ' order unity', fontsize=8, color='gray'); a.text(0.3, 0.04, 'very loose ', fontsize=8, color='gray', ha='right')
    a.text(alphas[-1], 0.70, 'dotted lines: welded limit', fontsize=8, ha='right', va='top')
    sec = a.secondary_xaxis('top', functions=(lambda x: e0_meV / np.maximum(x, 1e-9), lambda e: e0_meV / np.maximum(e, 1e-9)))
    sec.set_xlabel(r'phonon energy (meV) if $\alpha=1$ at %.1f meV ($\approx600$ GHz) and $K$ is frequency independent' % e0_meV, fontsize=8.5)
    sec.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50]); sec.set_xticklabels(['0.2', '0.5', '1', '2', '5', '10', '20', '50'])
    a.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    for e in ('png', 'pdf'): fig.savefig(os.path.join(outdir, 'interface_vs_alpha.' + e), dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--metal', default='Ta', choices=list(pim.MATERIALS)); ap.add_argument('--outdir', default='.'); ap.add_argument('--materials', default=None, help='optional JSON materials file'); ap.add_argument('--n', type=int, default=600)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    if a.materials: pim.load_materials(a.materials)
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    fig_concept(a.metal, a.outdir); fig_angular(a.metal, a.outdir, a.n); fig_vs_alpha(a.outdir, a.n)

if __name__ == '__main__':
    main()

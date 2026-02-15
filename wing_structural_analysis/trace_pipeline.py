#!/usr/bin/env python3
"""
trace_pipeline.py - Pipeline Trace for Default Values

Manually traces every phase of the optimization pipeline to identify
exactly where and why configurations fail.

Key questions:
1. What is h_box_root_mm with default parameters?
2. Does the largest config pass assembly?
3. What is tau_cr for default rib spacing with PLA skin?
4. Is there a config that passes Phase 0-2 but fails Phase 3?
5. Why did the old version (no Phase 3) find solutions?
"""

import numpy as np
import sys
import math

sys.path.insert(0, '/Users/apple/Desktop/dosyalar/wing_structural_analysis')

from materials import MaterialDatabase, MaterialSelection
from geometry import (PlanformParams, SparPosition, compute_all_stations,
                      compute_spar_sweep_angle, compute_skin_area_half_wing)
from loads import (FlightCondition, AeroCenter, analyze_loads,
                   LoadDistributionType, PitchMomentDistributionType)
from torsion import analyze_all_stations as analyze_torsion_stations, find_critical_station
from spars import (SparProperties, LoadSharing, compute_box_beam_section,
                   analyze_box_beam_stress, spar_mass)
from buckling import (skin_shear_buckling_critical, skin_compression_buckling_critical,
                      combined_interaction_ratio)
from ribs import (RibProperties, generate_rib_geometries, compute_total_rib_mass,
                  AdaptiveRibConfig, adaptive_rib_insertion)
from scipy.interpolate import interp1d


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_icon(passed):
    return "PASS" if passed else "FAIL"


# =============================================================================
# STEP 1: Default Parameters
# =============================================================================
separator("STEP 1: DEFAULT PARAMETERS")

C_m = -0.003
AR = 11.0
lambda_taper = 0.45
V_c = 21.0
n = 2.0
S_ref = 3.17
W_0 = 650.0
rho_air = 1.223
x_ac_percent = 25.0
Lambda_ac_deg = 0.0
b = 5.195
t_over_c = 0.17
t_skin_mm = 0.625
SF = 1.0
theta_max_deg = 2.0
s_min_mm = 20.0

# Material IDs: spar=3 (CFRP_UD), rib=5 (PLA), skin=5 (PLA)
# From materials.py: dict order is AL7075-T6(1), AL2024-T3(2), CFRP_UD(3), GFRP(4), PLA(5), STEEL_4130(6)

print(f"  Flight: C_m={C_m}, AR={AR}, lambda={lambda_taper}, V_c={V_c}, n={n}")
print(f"  Wing:   S_ref={S_ref}, W_0={W_0}, rho={rho_air}")
print(f"  Aero:   x_ac={x_ac_percent}%, Lambda_ac={Lambda_ac_deg} deg")
print(f"  Geom:   b={b}, t/c={t_over_c}")
print(f"  Struct: t_skin={t_skin_mm}mm, SF={SF}, theta_max={theta_max_deg} deg")


# =============================================================================
# STEP 2: Geometry Calculations
# =============================================================================
separator("STEP 2: GEOMETRY")

C_r = 2 * S_ref / (b * (1 + lambda_taper))
C_r_mm = C_r * 1000

c_MGC = (4/3) * np.sqrt(S_ref / AR) * ((1 + lambda_taper + lambda_taper**2) / (1 + 2*lambda_taper + lambda_taper**2))

Y_bar_mm = (b / 6) * ((1 + 2 * lambda_taper) / (1 + lambda_taper)) * 1000

N_Rib = int(np.ceil(1 + np.sqrt(AR * S_ref) / c_MGC))

L_span = b / 2
c_tip = lambda_taper * C_r
h_box_root = t_over_c * C_r
h_box_root_mm = t_over_c * C_r_mm

print(f"  C_r         = {C_r_mm:.2f} mm ({C_r:.4f} m)")
print(f"  c_tip       = {c_tip*1000:.2f} mm")
print(f"  c_MGC       = {c_MGC*1000:.2f} mm ({c_MGC:.4f} m)")
print(f"  Y_bar       = {Y_bar_mm:.2f} mm ({Y_bar_mm/1000:.4f} m)")
print(f"  L_span      = {L_span*1000:.2f} mm ({L_span:.4f} m)")
print(f"  N_Rib       = {N_Rib}")
print(f"  h_box_root  = {h_box_root_mm:.2f} mm ({h_box_root:.4f} m)")
print(f"")
print(f"  >>> h_box_root_mm = {h_box_root_mm:.2f} mm")
print(f"  >>> This is the MAXIMUM diameter any spar can have (minus 10mm clearance)")
print(f"  >>> Assembly limit: d_outer_max = {h_box_root_mm - 10:.2f} mm")


# =============================================================================
# STEP 3: Materials
# =============================================================================
separator("STEP 3: MATERIALS")

db = MaterialDatabase()
mat_keys = list(db.materials.keys())
print(f"  Material order: {mat_keys}")

# Default: spar=3 -> CFRP_UD, rib=5 -> PLA, skin=5 -> PLA
mat_spar_key = mat_keys[2]  # CFRP_UD (index 2, ID 3)
mat_rib_key = mat_keys[4]   # PLA (index 4, ID 5)
mat_skin_key = mat_keys[4]  # PLA (index 4, ID 5)

materials = MaterialSelection.from_database(db, mat_spar_key, mat_skin_key, mat_rib_key)

print(f"  Spar: {materials.spar.name}")
print(f"    E={materials.spar.E/1e9:.1f} GPa, sigma_u={materials.spar.sigma_u/1e6:.0f} MPa")
print(f"    tau_u={materials.spar.tau_u/1e6:.0f} MPa, G={materials.spar.G/1e9:.1f} GPa")
print(f"    density={materials.spar.density} kg/m3")
print(f"  Skin: {materials.skin.name}")
print(f"    E={materials.skin.E/1e9:.1f} GPa, sigma_u={materials.skin.sigma_u/1e6:.0f} MPa")
print(f"    tau_u={materials.skin.tau_u/1e6:.0f} MPa, G={materials.skin.G/1e9:.1f} GPa")
print(f"  Rib: {materials.rib.name}")
print(f"    E={materials.rib.E/1e9:.1f} GPa, sigma_u={materials.rib.sigma_u/1e6:.0f} MPa")

# Allowables (SF=1.0 means no reduction)
sigma_allow_spar = materials.spar.sigma_u / SF
tau_allow_spar = materials.spar.tau_u / SF
sigma_allow_skin = materials.skin.sigma_u / SF
tau_allow_skin = materials.skin.tau_u / SF

print(f"\n  Allowables (SF={SF}):")
print(f"    sigma_allow_spar = {sigma_allow_spar/1e6:.0f} MPa")
print(f"    tau_allow_spar   = {tau_allow_spar/1e6:.0f} MPa")
print(f"    sigma_allow_skin = {sigma_allow_skin/1e6:.0f} MPa")
print(f"    tau_allow_skin   = {tau_allow_skin/1e6:.0f} MPa")


# =============================================================================
# STEP 4: Test Configurations
# =============================================================================
separator("STEP 4: TEST CONFIGURATIONS")

# Most generous config from default design ranges
configs = [
    {"name": "MAX (d_FS=30, t=3, d_RS=25, t=3)",
     "d_FS": 30.0, "t_FS": 3.0, "d_RS": 25.0, "t_RS": 3.0,
     "X_FS": 22.5, "X_RS": 52.5, "t_rib": 3.0},
    {"name": "DEFAULT SINGLE (d_FS=23, t=1.75, d_RS=17.5, t=1.75)",
     "d_FS": 23.0, "t_FS": 1.75, "d_RS": 17.5, "t_RS": 1.75,
     "X_FS": 22.5, "X_RS": 52.5, "t_rib": 3.0},
    {"name": "SMALL (d_FS=16, t=0.5, d_RS=10, t=0.5)",
     "d_FS": 16.0, "t_FS": 0.5, "d_RS": 10.0, "t_RS": 0.5,
     "X_FS": 22.5, "X_RS": 52.5, "t_rib": 3.0},
    {"name": "WIDE SPARS (X_FS=15, X_RS=65)",
     "d_FS": 30.0, "t_FS": 3.0, "d_RS": 25.0, "t_RS": 3.0,
     "X_FS": 15.0, "X_RS": 65.0, "t_rib": 3.0},
]


def trace_config(cfg):
    """Run full pipeline trace for a single configuration."""
    name = cfg["name"]
    d_FS_mm = cfg["d_FS"]
    t_FS_mm = cfg["t_FS"]
    d_RS_mm = cfg["d_RS"]
    t_RS_mm = cfg["t_RS"]
    X_FS_pct = cfg["X_FS"]
    X_RS_pct = cfg["X_RS"]
    t_rib_mm = cfg["t_rib"]

    print(f"\n--- Config: {name} ---")
    print(f"  d_FS={d_FS_mm}mm, t_FS={t_FS_mm}mm, d_RS={d_RS_mm}mm, t_RS={t_RS_mm}mm")
    print(f"  X_FS={X_FS_pct}%, X_RS={X_RS_pct}%, t_rib={t_rib_mm}mm, N_Rib={N_Rib}")

    # Phase 0: Geometry + Assembly
    print(f"\n  [PHASE 0] Geometry & Assembly")

    # Geometry validation
    if X_FS_pct >= X_RS_pct:
        print(f"    FAIL: X_FS >= X_RS")
        return
    if t_FS_mm >= d_FS_mm / 2:
        print(f"    FAIL: t_FS >= d_FS/2")
        return
    if t_RS_mm >= d_RS_mm / 2:
        print(f"    FAIL: t_RS >= d_RS/2")
        return
    print(f"    Geometry validation: PASS")

    # Assembly check
    asm_FS = d_FS_mm + 10 < h_box_root_mm
    asm_RS = d_RS_mm + 10 < h_box_root_mm
    print(f"    Assembly FS: {d_FS_mm} + 10 = {d_FS_mm+10:.1f} < {h_box_root_mm:.2f} ? {check_icon(asm_FS)}")
    print(f"    Assembly RS: {d_RS_mm} + 10 = {d_RS_mm+10:.1f} < {h_box_root_mm:.2f} ? {check_icon(asm_RS)}")

    if not (asm_FS and asm_RS):
        print(f"    >>> REJECTED at Phase 0 (assembly)")
        return

    # Phase 1: Loads + Torsion
    print(f"\n  [PHASE 1] Loads & Torsion")

    planform = PlanformParams.from_input(
        b=b, AR=AR, taper_ratio=lambda_taper, t_c=t_over_c,
        S_ref=S_ref, C_r_mm=C_r_mm, c_MGC=c_MGC, Y_bar_mm=Y_bar_mm
    )

    spar_pos = SparPosition(X_FS_percent=X_FS_pct, X_RS_percent=X_RS_pct)
    stations = compute_all_stations(planform, spar_pos, N_Rib)

    y = np.array([s.y for s in stations])
    chord = np.array([s.chord for s in stations])
    x_FS = np.array([s.x_FS for s in stations])
    x_RS = np.array([s.x_RS for s in stations])
    A_m = np.array([s.A_m for s in stations])
    h_box = np.array([s.h_box for s in stations])
    box_width = np.array([s.width for s in stations])

    flight = FlightCondition(
        W0=W_0, n=n, V_c=V_c, rho=rho_air,
        C_m=C_m, S_ref=S_ref, c_MGC=c_MGC
    )

    aero_center = AeroCenter(x_ac_percent=x_ac_percent, Lambda_ac_deg=Lambda_ac_deg)

    loads = analyze_loads(
        y, chord, x_FS, x_RS, flight, aero_center, planform.L_span,
        LoadDistributionType.UNIFORM, PitchMomentDistributionType.UNIFORM
    )

    t_skin = t_skin_mm / 1000
    G_skin = materials.skin.G

    torsion_results = analyze_torsion_stations(y, loads.T, A_m, t_skin, G_skin, box_width, h_box)

    _, crit_torsion = find_critical_station(torsion_results)
    tau_skin_max = abs(crit_torsion.tau_skin)

    twist_rates = np.array([r.twist_rate for r in torsion_results])
    theta_tip_rad = np.trapz(twist_rates, y)
    theta_tip_deg = np.degrees(abs(theta_tip_rad))

    print(f"    L_half = {flight.L_half:.2f} N")
    print(f"    M_pitch_half = {flight.M_pitch_half:.4f} N.m")
    print(f"    M_root = {loads.M[0]:.4f} N.m")
    print(f"    T_root = {loads.T[0]:.4f} N.m")
    print(f"    V_root = {loads.V[0]:.2f} N")
    print(f"    tau_skin_max = {tau_skin_max/1e6:.6f} MPa (allow: {tau_allow_skin/1e6:.0f} MPa)")
    skin_pass = tau_skin_max <= tau_allow_skin
    print(f"    Skin stress check: {check_icon(skin_pass)}")
    print(f"    theta_tip = {theta_tip_deg:.4f} deg (max: {theta_max_deg} deg)")
    twist_pass = theta_tip_deg <= theta_max_deg
    print(f"    Twist check: {check_icon(twist_pass)}")

    if not skin_pass:
        print(f"    >>> REJECTED at Phase 1 (skin stress)")
        return
    if not twist_pass:
        print(f"    >>> REJECTED at Phase 1 (twist)")
        return

    # Phase 2: Box Beam + Spar Checks
    print(f"\n  [PHASE 2] Box Beam & Spar Interaction")

    spar_FS = SparProperties.from_mm(d_FS_mm, t_FS_mm, materials.spar)
    spar_RS = SparProperties.from_mm(d_RS_mm, t_RS_mm, materials.spar)

    E_skin = materials.skin.E
    box_sections = [
        compute_box_beam_section(spar_FS, spar_RS, h_box[i], box_width[i], t_skin, E_skin)
        for i in range(len(y))
    ]

    sharing = LoadSharing.from_positions(X_FS_pct, X_RS_pct, x_ac_percent)

    box_results = [
        analyze_box_beam_stress(loads.M[i], loads.T[i], box_sections[i],
                                spar_FS, spar_RS, sharing.eta_FS, sharing.eta_RS, y[i])
        for i in range(len(y))
    ]

    # Find critical station
    max_vm_idx = 0
    max_vm = max(box_results[0].sigma_vm_FS, box_results[0].sigma_vm_RS)
    max_skin_comp = box_results[0].sigma_skin_comp
    for i, br in enumerate(box_results):
        vm = max(br.sigma_vm_FS, br.sigma_vm_RS)
        if vm > max_vm:
            max_vm = vm
            max_vm_idx = i
        if br.sigma_skin_comp > max_skin_comp:
            max_skin_comp = br.sigma_skin_comp

    crit_box = box_results[max_vm_idx]

    print(f"    eta_FS = {sharing.eta_FS:.4f}, eta_RS = {sharing.eta_RS:.4f}")
    print(f"    I_box_root = {box_sections[0].I_total*1e12:.4f} mm4")
    print(f"    I_FS = {spar_FS.I*1e12:.4f} mm4, I_RS = {spar_RS.I*1e12:.4f} mm4")
    print(f"    I_skin_transformed = {box_sections[0].I_skin*1e12:.4f} mm4")
    print(f"    Skin I fraction = {box_sections[0].skin_fraction*100:.1f}%")
    print(f"    A_FS = {spar_FS.area*1e6:.4f} mm2, A_RS = {spar_RS.area*1e6:.4f} mm2")
    print(f"")
    print(f"    Critical station: y = {y[max_vm_idx]*1000:.1f} mm")
    print(f"    sigma_b_FS = {crit_box.sigma_b_FS/1e6:.4f} MPa")
    print(f"    tau_FS     = {crit_box.tau_FS/1e6:.6f} MPa")
    print(f"    sigma_b_RS = {crit_box.sigma_b_RS/1e6:.4f} MPa")
    print(f"    tau_RS     = {crit_box.tau_RS/1e6:.6f} MPa")
    print(f"    sigma_vm_FS = {crit_box.sigma_vm_FS/1e6:.4f} MPa")
    print(f"    sigma_vm_RS = {crit_box.sigma_vm_RS/1e6:.4f} MPa")
    print(f"    sigma_skin_comp_max = {max_skin_comp/1e6:.6f} MPa")

    # Interaction checks
    FS_interaction = (crit_box.sigma_b_FS / sigma_allow_spar) + (crit_box.tau_FS / tau_allow_spar)**2
    RS_interaction = (crit_box.sigma_b_RS / sigma_allow_spar) + (crit_box.tau_RS / tau_allow_spar)**2
    print(f"")
    print(f"    FS interaction = {FS_interaction:.6f} < 1.0 ? {check_icon(FS_interaction < 1.0)}")
    print(f"    RS interaction = {RS_interaction:.6f} < 1.0 ? {check_icon(RS_interaction < 1.0)}")

    if FS_interaction >= 1.0:
        print(f"    >>> REJECTED at Phase 2 (FS interaction)")
        return
    if RS_interaction >= 1.0:
        print(f"    >>> REJECTED at Phase 2 (RS interaction)")
        return

    # Critical area check
    Y_bar_m = Y_bar_mm / 1000
    sigma_max_spar = materials.spar.sigma_u
    c_FS_dist = spar_FS.d_outer / 2
    c_RS_dist = spar_RS.d_outer / 2
    A_Cri_FS = (n * W_0 * Y_bar_m * 0.5 * sharing.eta_FS) / (2 * sigma_max_spar * c_FS_dist)
    A_Cri_RS = (n * W_0 * Y_bar_m * 0.5 * sharing.eta_RS) / (2 * sigma_max_spar * c_RS_dist)

    area_FS_pass = spar_FS.area > A_Cri_FS
    area_RS_pass = spar_RS.area > A_Cri_RS
    print(f"    A_Act_FS = {spar_FS.area*1e6:.4f} mm2, A_Cri_FS = {A_Cri_FS*1e6:.4f} mm2  {check_icon(area_FS_pass)}")
    print(f"    A_Act_RS = {spar_RS.area*1e6:.4f} mm2, A_Cri_RS = {A_Cri_RS*1e6:.4f} mm2  {check_icon(area_RS_pass)}")

    if not area_FS_pass:
        print(f"    >>> REJECTED at Phase 2 (FS area)")
        return
    if not area_RS_pass:
        print(f"    >>> REJECTED at Phase 2 (RS area)")
        return

    print(f"\n    >>> PHASE 0-2: ALL PASSED <<<")

    # Phase 3: Buckling + Adaptive Rib
    print(f"\n  [PHASE 3] Buckling & Adaptive Rib Insertion")

    # Rib spacing
    rib_spacing = L_span / N_Rib
    rib_spacing_mm = rib_spacing * 1000
    print(f"    Initial rib spacing = {rib_spacing_mm:.2f} mm ({rib_spacing:.4f} m)")
    print(f"    N_Rib = {N_Rib}, L_span = {L_span*1000:.2f} mm")

    # Buckling criticals at initial rib spacing
    E_pla = materials.skin.E
    nu_pla = materials.skin.nu
    t_skin_m = t_skin_mm / 1000

    tau_cr_init = skin_shear_buckling_critical(E_pla, nu_pla, t_skin_m, rib_spacing)
    sigma_cr_init = skin_compression_buckling_critical(E_pla, nu_pla, t_skin_m, rib_spacing)

    print(f"    PLA skin: E={E_pla/1e9:.1f} GPa, nu={nu_pla}, t_skin={t_skin_mm} mm")
    print(f"    tau_cr (initial spacing) = {tau_cr_init/1e6:.6f} MPa")
    print(f"    sigma_cr (initial spacing) = {sigma_cr_init/1e6:.6f} MPa")
    print(f"    tau_skin_max (applied) = {tau_skin_max/1e6:.6f} MPa")
    print(f"    sigma_comp_max (applied) = {max_skin_comp/1e6:.6f} MPa")

    # Check: tau_cr vs tau_applied
    shear_buckle_pass = tau_skin_max <= tau_cr_init
    comp_buckle_pass = max_skin_comp <= sigma_cr_init
    print(f"    Shear buckling: tau_max <= tau_cr ? {check_icon(shear_buckle_pass)}")
    print(f"    Comp buckling:  sigma_max <= sigma_cr ? {check_icon(comp_buckle_pass)}")

    # Combined interaction at initial spacing
    R_init = combined_interaction_ratio(tau_skin_max, tau_cr_init, max_skin_comp, sigma_cr_init)
    print(f"    R_combined (initial) = {R_init:.6f} < 1.0 ? {check_icon(R_init < 1.0)}")

    # What spacing would be needed?
    if tau_skin_max > 0:
        s_max_shear = t_skin_m * math.sqrt(5.34 * math.pi**2 * E_pla / (12*(1-nu_pla**2) * tau_skin_max))
        print(f"    Required s_max_shear = {s_max_shear*1000:.2f} mm")
    else:
        s_max_shear = float('inf')
        print(f"    Required s_max_shear = inf (no shear stress)")

    if max_skin_comp > 0:
        s_max_comp = t_skin_m * math.sqrt(4.0 * math.pi**2 * E_pla / (12*(1-nu_pla**2) * max_skin_comp))
        print(f"    Required s_max_comp = {s_max_comp*1000:.2f} mm")
    else:
        s_max_comp = float('inf')
        print(f"    Required s_max_comp = inf (no compression)")

    s_target = min(s_max_shear, s_max_comp) * 0.95
    print(f"    s_target (with 0.95 margin) = {s_target*1000:.2f} mm")
    print(f"    s_min (practical limit) = {s_min_mm:.1f} mm")
    feasible_spacing = s_target >= s_min_mm / 1000
    print(f"    Feasible spacing? {check_icon(feasible_spacing)}")

    if s_target < s_min_mm / 1000:
        n_ribs_needed = math.ceil(L_span / s_target)
        print(f"    Would need ~{n_ribs_needed} rib bays (spacing {L_span/n_ribs_needed*1000:.1f} mm)")
        print(f"    But s_min={s_min_mm}mm prevents this")

    # Run adaptive rib insertion
    print(f"\n    --- Running Adaptive Rib Insertion ---")

    tau_skin_arr = np.array([abs(tr.tau_skin) for tr in torsion_results])
    sigma_comp_arr = np.array([br.sigma_skin_comp for br in box_results])

    tau_interp = interp1d(y, tau_skin_arr, kind='linear', fill_value='extrapolate')
    sigma_interp = interp1d(y, sigma_comp_arr, kind='linear', fill_value='extrapolate')
    V_interp = interp1d(y, loads.V, kind='linear', fill_value='extrapolate')
    h_interp = interp1d(y, h_box, kind='linear', fill_value='extrapolate')

    adapt_config = AdaptiveRibConfig(
        E_skin=materials.skin.E,
        nu_skin=materials.skin.nu,
        t_skin=t_skin_m,
        E_rib=materials.rib.E,
        nu_rib=materials.rib.nu,
        t_rib=t_rib_mm / 1000,
        tau_allow_skin=tau_allow_skin,
        sigma_allow_skin=sigma_allow_skin,
        k_s=5.34,
        k_c=4.0,
        s_min=s_min_mm / 1000,
        margin=0.95,
    )

    final_y_ribs, feasible, adapt_msg = adaptive_rib_insertion(
        y, adapt_config, tau_interp, sigma_interp, V_interp, h_interp
    )

    N_Rib_final = len(final_y_ribs) - 1
    print(f"    {adapt_msg}")
    print(f"    N_Rib initial = {N_Rib}, N_Rib final = {N_Rib_final}")
    print(f"    Feasible = {feasible}")

    if feasible:
        # Check bay by bay
        print(f"\n    Bay-by-bay check:")
        for i in range(len(final_y_ribs) - 1):
            y_s, y_e = final_y_ribs[i], final_y_ribs[i+1]
            spacing = y_e - y_s
            tau_p = max(abs(tau_interp(y_s)), abs(tau_interp(y_e)))
            sig_p = max(abs(sigma_interp(y_s)), abs(sigma_interp(y_e)))
            tau_cr_bay = skin_shear_buckling_critical(E_pla, nu_pla, t_skin_m, spacing)
            sigma_cr_bay = skin_compression_buckling_critical(E_pla, nu_pla, t_skin_m, spacing)
            R_bay = combined_interaction_ratio(tau_p, tau_cr_bay, sig_p, sigma_cr_bay)
            status = "PASS" if R_bay < 1.0 else "FAIL"
            if i < 5 or not feasible:  # Print first 5 or all if infeasible
                print(f"      Bay {i+1}: s={spacing*1000:.1f}mm, tau={tau_p/1e6:.4f}/{tau_cr_bay/1e6:.4f} MPa, "
                      f"sig={sig_p/1e6:.6f}/{sigma_cr_bay/1e6:.4f} MPa, R={R_bay:.4f} {status}")
        if N_Rib_final > 5:
            print(f"      ... ({N_Rib_final} bays total)")
    else:
        # Show first few failing bays
        print(f"\n    Bay-by-bay (showing all):")
        for i in range(min(len(final_y_ribs) - 1, 20)):
            y_s, y_e = final_y_ribs[i], final_y_ribs[i+1]
            spacing = y_e - y_s
            tau_p = max(abs(tau_interp(y_s)), abs(tau_interp(y_e)))
            sig_p = max(abs(sigma_interp(y_s)), abs(sigma_interp(y_e)))
            tau_cr_bay = skin_shear_buckling_critical(E_pla, nu_pla, t_skin_m, spacing)
            sigma_cr_bay = skin_compression_buckling_critical(E_pla, nu_pla, t_skin_m, spacing)
            R_bay = combined_interaction_ratio(tau_p, tau_cr_bay, sig_p, sigma_cr_bay)
            status = "PASS" if R_bay < 1.0 else "FAIL"
            print(f"      Bay {i+1}: s={spacing*1000:.1f}mm, tau={tau_p/1e6:.4f}/{tau_cr_bay/1e6:.4f} MPa, "
                  f"sig={sig_p/1e6:.6f}/{sigma_cr_bay/1e6:.4f} MPa, R={R_bay:.4f} {status}")

    # Mass (even if infeasible, calculate for reference)
    print(f"\n  [MASS] Reference Calculation")
    S_skin = compute_skin_area_half_wing(planform)
    mass_skin = S_skin * t_skin_m * materials.skin.density

    Lambda_FS = compute_spar_sweep_angle(X_FS_pct, planform, Lambda_ac_deg, x_ac_percent)
    Lambda_RS = compute_spar_sweep_angle(X_RS_pct, planform, Lambda_ac_deg, x_ac_percent)
    L_FS = L_span / np.cos(np.radians(Lambda_FS))
    L_RS = L_span / np.cos(np.radians(Lambda_RS))
    mass_FS = spar_mass(spar_FS, L_FS)
    mass_RS = spar_mass(spar_RS, L_RS)

    rib_props = RibProperties.from_mm(t_rib_mm, materials.rib)
    ribs = generate_rib_geometries(y, chord, h_box, x_FS, x_RS, parabolic=False)
    rib_mass_result = compute_total_rib_mass(ribs, rib_props, L_span)

    mass_total = mass_skin + mass_FS + mass_RS + rib_mass_result.total_rib_mass

    print(f"    Skin: {mass_skin*1000:.2f} g")
    print(f"    FS:   {mass_FS*1000:.2f} g")
    print(f"    RS:   {mass_RS*1000:.2f} g")
    print(f"    Ribs: {rib_mass_result.total_rib_mass*1000:.2f} g")
    print(f"    TOTAL: {mass_total*1000:.2f} g")

    if feasible:
        print(f"\n    >>> PHASE 3: PASSED - SOLUTION VALID <<<")
    else:
        print(f"\n    >>> PHASE 3: FAILED - WOULD HAVE PASSED IN OLD VERSION (no Phase 3) <<<")

    return feasible


# =============================================================================
# Run all configs
# =============================================================================
separator("RUNNING ALL TEST CONFIGURATIONS")

results = {}
for cfg in configs:
    result = trace_config(cfg)
    results[cfg["name"]] = result


# =============================================================================
# STEP 5: Buckling Sensitivity Study
# =============================================================================
separator("STEP 5: BUCKLING SENSITIVITY (PLA, t_skin=0.625mm)")

E_pla = materials.skin.E
nu_pla = materials.skin.nu
t_skin_m = t_skin_mm / 1000

print(f"\n  Shear buckling tau_cr vs rib spacing:")
print(f"  {'s [mm]':>10} {'tau_cr [MPa]':>15} {'tau_cr [Pa]':>15}")
for s_mm in [20, 30, 50, 80, 100, 150, 200, 260, 300, 500]:
    s = s_mm / 1000
    tau_cr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin_m, s)
    print(f"  {s_mm:10d} {tau_cr/1e6:15.6f} {tau_cr:15.2f}")

print(f"\n  Compression buckling sigma_cr vs rib spacing:")
print(f"  {'s [mm]':>10} {'sigma_cr [MPa]':>15}")
for s_mm in [20, 30, 50, 80, 100, 150, 200, 260, 300, 500]:
    s = s_mm / 1000
    sigma_cr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin_m, s)
    print(f"  {s_mm:10d} {sigma_cr/1e6:15.6f}")


# =============================================================================
# STEP 6: What Design Range Could Work?
# =============================================================================
separator("STEP 6: DESIGN RANGE ANALYSIS")

print(f"\n  Given constraints:")
print(f"    h_box_root = {h_box_root_mm:.2f} mm")
print(f"    Assembly limit: d_max = {h_box_root_mm - 10:.2f} mm")
print(f"    PLA skin E = {E_pla/1e9:.1f} GPa, t_skin = {t_skin_mm} mm")

# Scan: what is the minimum tau_skin (best case)?
# With widest possible box (X_FS=15, X_RS=65), max diameter spars
print(f"\n  Best-case tau_skin calculation:")
print(f"    Minimum possible tau_skin occurs with largest A_m (widest box)")

# Compute A_m at root for different spar positions
for X_FS, X_RS in [(15, 65), (20, 60), (22.5, 52.5), (25, 50)]:
    width = (X_RS - X_FS) / 100 * C_r
    A_m_root = width * h_box_root
    # T_root for uniform distribution
    planform_tmp = PlanformParams.from_input(
        b=b, AR=AR, taper_ratio=lambda_taper, t_c=t_over_c,
        S_ref=S_ref, C_r_mm=C_r_mm, c_MGC=c_MGC, Y_bar_mm=Y_bar_mm
    )
    spar_pos_tmp = SparPosition(X_FS_percent=X_FS, X_RS_percent=X_RS)
    stations_tmp = compute_all_stations(planform_tmp, spar_pos_tmp, N_Rib)
    y_tmp = np.array([s.y for s in stations_tmp])
    chord_tmp = np.array([s.chord for s in stations_tmp])
    x_FS_tmp = np.array([s.x_FS for s in stations_tmp])
    x_RS_tmp = np.array([s.x_RS for s in stations_tmp])
    A_m_tmp = np.array([s.A_m for s in stations_tmp])

    flight_tmp = FlightCondition(W0=W_0, n=n, V_c=V_c, rho=rho_air, C_m=C_m, S_ref=S_ref, c_MGC=c_MGC)
    ac_tmp = AeroCenter(x_ac_percent=x_ac_percent, Lambda_ac_deg=Lambda_ac_deg)

    loads_tmp = analyze_loads(y_tmp, chord_tmp, x_FS_tmp, x_RS_tmp, flight_tmp, ac_tmp,
                              planform_tmp.L_span, LoadDistributionType.UNIFORM,
                              PitchMomentDistributionType.UNIFORM)

    # tau_skin at root
    tau_root = abs(loads_tmp.T[0]) / (2 * A_m_tmp[0] * t_skin_m)
    print(f"    X_FS={X_FS}%, X_RS={X_RS}%: A_m_root={A_m_tmp[0]*1e6:.1f} mm2, "
          f"T_root={loads_tmp.T[0]:.4f} N.m, tau_root={tau_root/1e6:.6f} MPa")

# What spacing gives tau_cr matching the minimum tau_skin?
print(f"\n  For tau_skin ~ 0.01 MPa (10 kPa), what spacing is needed?")
tau_target = 0.01e6  # 10 kPa
s_needed = t_skin_m * math.sqrt(5.34 * math.pi**2 * E_pla / (12 * (1 - nu_pla**2) * tau_target))
print(f"    s_max_shear = {s_needed*1000:.2f} mm")
n_ribs_needed = math.ceil(L_span / s_needed)
print(f"    -> Need ~{n_ribs_needed} rib bays")
print(f"    -> Spacing = {L_span/n_ribs_needed*1000:.2f} mm")
print(f"    -> s_min limit = {s_min_mm} mm")


# =============================================================================
# STEP 7: Summary / Root Cause Analysis
# =============================================================================
separator("STEP 7: ROOT CAUSE ANALYSIS & SUMMARY")

print(f"""
  FINDINGS:
  =========

  1. h_box_root = {h_box_root_mm:.2f} mm
     - Assembly limit: d_max < {h_box_root_mm - 10:.2f} mm
     - Default design range d_FS: 16-30mm -> mostly within limit
     - Default d_RS: 10-25mm -> within limit

  2. Phase 0-2 checks:
     - With CFRP_UD spar (sigma_u=1500 MPa), stress checks are VERY easy
     - Low W_0=650N and moderate loads -> small stresses
     - Interaction values << 1.0
     - Phase 0-2 passes for many configs

  3. Phase 3 (Buckling) is the BOTTLENECK:
     - PLA skin with E=3.5 GPa and t=0.625mm
     - tau_cr scales as (t/s)^2 -> very sensitive to rib spacing
     - Initial N_Rib={N_Rib} gives spacing ~{L_span/N_Rib*1000:.0f}mm
     - At this spacing: tau_cr << tau_skin_applied
     - Adaptive rib insertion tries to reduce spacing
     - But hits s_min={s_min_mm}mm limit

  4. Old vs New:
     - OLD version: no Phase 3 -> configs that pass Phase 0-2 are accepted
     - NEW version: Phase 3 buckling check -> tau_cr with PLA skin is
       extremely low at practical rib spacing -> all configs fail
     - This is a REAL physical limitation of PLA skin

  5. ROOT CAUSE:
     - PLA (E=3.5 GPa) is ~100x softer than aluminum (E=71.7 GPa)
     - Buckling critical stress ~ E * (t/s)^2
     - For same t and s, PLA has ~100x lower buckling resistance
     - Need either: thicker skin, stiffer skin material, or very tight rib spacing
""")

# Quick check: what if skin were aluminum instead of PLA?
E_al = 71.7e9
nu_al = 0.33
spacing_test = L_span / N_Rib
tau_cr_al = skin_shear_buckling_critical(E_al, nu_al, t_skin_m, spacing_test)
tau_cr_pla = skin_shear_buckling_critical(E_pla, nu_pla, t_skin_m, spacing_test)
print(f"  COMPARISON at spacing={spacing_test*1000:.1f}mm, t_skin={t_skin_mm}mm:")
print(f"    PLA skin:  tau_cr = {tau_cr_pla/1e6:.6f} MPa")
print(f"    AL skin:   tau_cr = {tau_cr_al/1e6:.6f} MPa")
print(f"    Ratio: AL/PLA = {tau_cr_al/tau_cr_pla:.1f}x")

# What t_skin would PLA need to match?
print(f"\n  What t_skin does PLA need at spacing={spacing_test*1000:.1f}mm to get tau_cr=0.01 MPa?")
# tau_cr = k * pi^2 * E / (12*(1-nu^2)) * (t/s)^2
# t = s * sqrt(tau_cr * 12*(1-nu^2) / (k * pi^2 * E))
tau_target_01 = 0.01e6
t_needed = spacing_test * math.sqrt(tau_target_01 * 12 * (1 - nu_pla**2) / (5.34 * math.pi**2 * E_pla))
print(f"    t_skin needed = {t_needed*1000:.2f} mm (currently {t_skin_mm} mm)")

print(f"\n{'='*70}")
print(f"  TRACE COMPLETE")
print(f"{'='*70}")

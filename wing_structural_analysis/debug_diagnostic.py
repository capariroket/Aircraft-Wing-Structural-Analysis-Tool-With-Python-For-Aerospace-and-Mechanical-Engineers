#!/usr/bin/env python3
"""
debug_diagnostic.py - Diagnostic Script for Wing Structural Analysis

Investigates why default values produce NO valid solutions in grid search.
Runs a single default configuration through all optimization phases
and reports exactly where and why each check fails.

Author: Debugger Agent
Date: 2026-02-15
"""

import numpy as np
import math
import sys
import os

# Ensure project directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from materials import MaterialDatabase, MaterialSelection
from geometry import (PlanformParams, SparPosition, compute_all_stations,
                      compute_spar_sweep_angle, compute_skin_area_half_wing,
                      chord_at_station as chord_at_station_val)
from loads import (FlightCondition, AeroCenter, analyze_loads,
                   LoadDistributionType, PitchMomentDistributionType)
from torsion import analyze_all_stations as analyze_torsion_stations, find_critical_station
from spars import (SparProperties, LoadSharing, spar_mass,
                   BoxBeamSection, compute_box_beam_section, analyze_box_beam_stress,
                   tip_deflection_from_moment)
from ribs import (RibProperties, generate_rib_geometries, compute_total_rib_mass,
                  AdaptiveRibConfig, adaptive_rib_insertion)
from buckling import (skin_shear_buckling_critical, skin_compression_buckling_critical,
                      combined_interaction_ratio, compute_s_max_shear, compute_s_max_compression)
from scipy.interpolate import interp1d


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_result(name: str, value, limit, operator="<", unit=""):
    """Print a check result with PASS/FAIL."""
    if operator == "<":
        passed = value < limit
        symbol = "<"
    elif operator == "<=":
        passed = value <= limit
        symbol = "<="
    elif operator == ">":
        passed = value > limit
        symbol = ">"
    elif operator == ">=":
        passed = value >= limit
        symbol = ">="
    else:
        passed = False
        symbol = "?"

    status = "PASS" if passed else "*** FAIL ***"
    if unit:
        print(f"  {name}: {value:.6f} {unit} {symbol} {limit:.6f} {unit} -> {status}")
    else:
        print(f"  {name}: {value:.6f} {symbol} {limit:.6f} -> {status}")
    return passed


def main():
    banner("DIAGNOSTIC: Why No Solutions with Default Values?")

    # =========================================================================
    # STEP 1: Setup - same as main.py DEFAULTS
    # =========================================================================
    banner("STEP 1: Default Parameters")

    # Flight science
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

    # Geometry
    b = 5.195
    t_over_c = 0.17

    # Structural
    t_skin_mm = 0.625
    SF = 1.0
    theta_max_deg = 2.0
    s_min_mm = 20.0
    rib_profile = 1  # Rectangular

    # Default single values
    t_rib_mm = 3.0
    X_FS_percent = 22.5
    X_RS_percent = 52.5
    d_FS_outer_mm = 23.0
    d_RS_outer_mm = 17.5
    t_FS_mm = 1.75
    t_RS_mm = 1.75

    # Material IDs -> keys
    mat_spar_id = 3   # Carbon Fiber (CFRP_UD)
    mat_rib_id = 5     # PLA
    mat_skin_id = 5    # PLA

    # Calculate derived geometry
    C_r = 2 * S_ref / (b * (1 + lambda_taper))
    C_r_mm = C_r * 1000
    c_MGC = (4/3) * np.sqrt(S_ref / AR) * ((1 + lambda_taper + lambda_taper**2) / (1 + 2*lambda_taper + lambda_taper**2))
    Y_bar_mm = (b / 6) * ((1 + 2 * lambda_taper) / (1 + lambda_taper)) * 1000
    N_Rib = int(np.ceil(1 + np.sqrt(AR * S_ref) / c_MGC))
    L_span = b / 2
    c_tip = lambda_taper * C_r
    h_box_root = t_over_c * C_r
    h_box_root_mm = t_over_c * C_r_mm

    print(f"  C_r = {C_r_mm:.2f} mm ({C_r:.4f} m)")
    print(f"  c_MGC = {c_MGC*1000:.2f} mm ({c_MGC:.4f} m)")
    print(f"  Y_bar = {Y_bar_mm:.2f} mm")
    print(f"  N_Rib = {N_Rib}")
    print(f"  L_span = {L_span*1000:.2f} mm ({L_span:.4f} m)")
    print(f"  c_tip = {c_tip*1000:.2f} mm")
    print(f"  h_box_root = {h_box_root*1000:.2f} mm ({h_box_root:.4f} m)")
    print(f"  Bay spacing (uniform) = {L_span/N_Rib*1000:.2f} mm")

    # Materials
    db = MaterialDatabase()
    mat_keys = list(db.materials.keys())
    mat_spar_key = mat_keys[mat_spar_id - 1]
    mat_rib_key = mat_keys[mat_rib_id - 1]
    mat_skin_key = mat_keys[mat_skin_id - 1]

    materials = MaterialSelection.from_database(db, mat_spar_key, mat_skin_key, mat_rib_key)

    print(f"\n  Spar material: {materials.spar.name}")
    print(f"    E = {materials.spar.E/1e9:.1f} GPa, sigma_u = {materials.spar.sigma_u/1e6:.0f} MPa, tau_u = {materials.spar.tau_u/1e6:.0f} MPa")
    print(f"    G = {materials.spar.G/1e9:.1f} GPa, density = {materials.spar.density} kg/m3")
    print(f"  Skin material: {materials.skin.name}")
    print(f"    E = {materials.skin.E/1e9:.1f} GPa, sigma_u = {materials.skin.sigma_u/1e6:.0f} MPa, tau_u = {materials.skin.tau_u/1e6:.0f} MPa")
    print(f"    G = {materials.skin.G/1e9:.2f} GPa, nu = {materials.skin.nu}")
    print(f"  Rib material: {materials.rib.name}")
    print(f"    E = {materials.rib.E/1e9:.1f} GPa, sigma_u = {materials.rib.sigma_u/1e6:.0f} MPa, tau_u = {materials.rib.tau_u/1e6:.0f} MPa")

    # Allowables
    sigma_allow_spar = materials.spar.sigma_u / SF
    tau_allow_spar = materials.spar.tau_u / SF
    sigma_allow_skin = materials.skin.sigma_u / SF
    tau_allow_skin = materials.skin.tau_u / SF

    print(f"\n  SF = {SF}")
    print(f"  sigma_allow_spar = {sigma_allow_spar/1e6:.1f} MPa")
    print(f"  tau_allow_spar = {tau_allow_spar/1e6:.1f} MPa")
    print(f"  sigma_allow_skin = {sigma_allow_skin/1e6:.1f} MPa")
    print(f"  tau_allow_skin = {tau_allow_skin/1e6:.1f} MPa")

    # =========================================================================
    # STEP 2: Phase 0 - Geometry & Assembly
    # =========================================================================
    banner("PHASE 0: Geometry Validation & Assembly Check")

    # Geometry checks
    p0_pass = True

    print(f"  X_FS = {X_FS_percent}%, X_RS = {X_RS_percent}%")
    if X_FS_percent >= X_RS_percent:
        print(f"  *** FAIL: X_FS >= X_RS")
        p0_pass = False
    else:
        print(f"  X_FS < X_RS: PASS")

    print(f"\n  d_FS = {d_FS_outer_mm} mm, t_FS = {t_FS_mm} mm -> d_FS/2 = {d_FS_outer_mm/2} mm")
    if t_FS_mm >= d_FS_outer_mm / 2:
        print(f"  *** FAIL: t_FS >= d_FS/2")
        p0_pass = False
    else:
        print(f"  t_FS < d_FS/2: PASS")

    print(f"  d_RS = {d_RS_outer_mm} mm, t_RS = {t_RS_mm} mm -> d_RS/2 = {d_RS_outer_mm/2} mm")
    if t_RS_mm >= d_RS_outer_mm / 2:
        print(f"  *** FAIL: t_RS >= d_RS/2")
        p0_pass = False
    else:
        print(f"  t_RS < d_RS/2: PASS")

    # Assembly check
    print(f"\n  Assembly check: d_spar + 10 < h_box")
    print(f"  h_box_root = {h_box_root_mm:.2f} mm")

    fs_assembly = d_FS_outer_mm + 10 < h_box_root_mm
    rs_assembly = d_RS_outer_mm + 10 < h_box_root_mm
    print(f"  FS: {d_FS_outer_mm} + 10 = {d_FS_outer_mm + 10} < {h_box_root_mm:.2f} -> {'PASS' if fs_assembly else '*** FAIL ***'}")
    print(f"  RS: {d_RS_outer_mm} + 10 = {d_RS_outer_mm + 10} < {h_box_root_mm:.2f} -> {'PASS' if rs_assembly else '*** FAIL ***'}")

    if not fs_assembly or not rs_assembly:
        p0_pass = False

    print(f"\n  PHASE 0 RESULT: {'PASS' if p0_pass else '*** FAIL ***'}")

    if not p0_pass:
        print("\n  Phase 0 failed. Would be rejected at geometry/assembly stage.")
        print("  However, continuing to see all phases...")

    # =========================================================================
    # STEP 3: Phase 1 - Loads + Torsion -> Skin stress & Twist
    # =========================================================================
    banner("PHASE 1: Loads + Torsion -> Skin Stress & Twist")

    planform = PlanformParams.from_input(
        b=b, AR=AR, taper_ratio=lambda_taper, t_c=t_over_c,
        S_ref=S_ref, C_r_mm=C_r_mm, c_MGC=c_MGC, Y_bar_mm=Y_bar_mm
    )

    flight = FlightCondition(
        W0=W_0, n=n, V_c=V_c, rho=rho_air,
        C_m=C_m, S_ref=S_ref, c_MGC=c_MGC
    )

    aero_center = AeroCenter(x_ac_percent=x_ac_percent, Lambda_ac_deg=Lambda_ac_deg)

    spar_pos = SparPosition(X_FS_percent=X_FS_percent, X_RS_percent=X_RS_percent)
    stations = compute_all_stations(planform, spar_pos, N_Rib)

    y = np.array([s.y for s in stations])
    chord = np.array([s.chord for s in stations])
    x_FS = np.array([s.x_FS for s in stations])
    x_RS = np.array([s.x_RS for s in stations])
    A_m = np.array([s.A_m for s in stations])
    h_box = np.array([s.h_box for s in stations])
    box_width = np.array([s.width for s in stations])

    print(f"\n  Flight loads:")
    print(f"    n*W_0 = {n*W_0:.1f} N")
    print(f"    L_half = {flight.L_half:.1f} N")
    print(f"    q_inf = {flight.q_inf:.2f} Pa")
    print(f"    M_pitch_half = {flight.M_pitch_half:.4f} N.m")

    # Loads analysis
    loads = analyze_loads(
        y, chord, x_FS, x_RS, flight, aero_center, planform.L_span,
        LoadDistributionType.UNIFORM, PitchMomentDistributionType.UNIFORM
    )

    print(f"\n  Root reactions:")
    print(f"    V_root = {loads.V[0]:.2f} N")
    print(f"    M_root = {loads.M[0]:.4f} N.m")
    print(f"    T_root = {loads.T[0]:.6f} N.m")

    # Torsion
    t_skin = t_skin_mm / 1000
    G_skin = materials.skin.G

    torsion_results = analyze_torsion_stations(
        y, loads.T, A_m, t_skin, G_skin, box_width, h_box
    )

    _, crit_torsion = find_critical_station(torsion_results)
    tau_skin_max = abs(crit_torsion.tau_skin)

    twist_rates = np.array([r.twist_rate for r in torsion_results])
    theta_tip_rad = np.trapz(twist_rates, y)
    theta_tip_deg = np.degrees(abs(theta_tip_rad))

    print(f"\n  Torsion results:")
    print(f"    tau_skin_max = {tau_skin_max/1e6:.6f} MPa")
    print(f"    theta_tip = {theta_tip_deg:.4f} deg")

    p1_pass = True

    tau_check = check_result("Skin shear", tau_skin_max/1e6, tau_allow_skin/1e6, "<", "MPa")
    if not tau_check:
        p1_pass = False

    twist_check = check_result("Tip twist", theta_tip_deg, theta_max_deg, "<=", "deg")
    if not twist_check:
        p1_pass = False

    print(f"\n  PHASE 1 RESULT: {'PASS' if p1_pass else '*** FAIL ***'}")

    # =========================================================================
    # STEP 4: Phase 2 - Box beam -> Spar Interaction & Area
    # =========================================================================
    banner("PHASE 2: Box Beam -> Spar Interaction & Area Checks")

    mat_spar = materials.spar

    spar_FS = SparProperties.from_mm(d_FS_outer_mm, t_FS_mm, mat_spar)
    spar_RS = SparProperties.from_mm(d_RS_outer_mm, t_RS_mm, mat_spar)

    print(f"  Spar FS: d={d_FS_outer_mm}mm, t={t_FS_mm}mm, I={spar_FS.I*1e12:.2f} mm4, A={spar_FS.area*1e6:.2f} mm2")
    print(f"  Spar RS: d={d_RS_outer_mm}mm, t={t_RS_mm}mm, I={spar_RS.I*1e12:.2f} mm4, A={spar_RS.area*1e6:.2f} mm2")

    E_skin = materials.skin.E
    box_sections = [
        compute_box_beam_section(spar_FS, spar_RS, h_box[i], box_width[i], t_skin, E_skin)
        for i in range(len(y))
    ]

    sharing = LoadSharing.from_positions(X_FS_percent, X_RS_percent, x_ac_percent)

    print(f"\n  Load sharing:")
    print(f"    eta_FS = {sharing.eta_FS:.4f} ({sharing.eta_FS*100:.1f}%)")
    print(f"    eta_RS = {sharing.eta_RS:.4f} ({sharing.eta_RS*100:.1f}%)")

    box_results = [
        analyze_box_beam_stress(loads.M[i], loads.T[i], box_sections[i],
                                spar_FS, spar_RS, sharing.eta_FS, sharing.eta_RS, y[i])
        for i in range(len(y))
    ]

    # Find critical station
    max_vm = 0
    max_vm_idx = 0
    max_skin_comp = 0
    for i, br in enumerate(box_results):
        vm = max(br.sigma_vm_FS, br.sigma_vm_RS)
        if vm > max_vm:
            max_vm = vm
            max_vm_idx = i
        if br.sigma_skin_comp > max_skin_comp:
            max_skin_comp = br.sigma_skin_comp

    crit_box = box_results[max_vm_idx]

    print(f"\n  Box beam analysis (root station):")
    print(f"    I_box_root = {box_sections[0].I_total*1e12:.2f} mm4")
    print(f"    n_skin (modular ratio) = {box_sections[0].n_skin:.4f}")
    print(f"    I_FS = {spar_FS.I*1e12:.2f} mm4")
    print(f"    I_RS = {spar_RS.I*1e12:.2f} mm4")
    print(f"    I_skin = {box_sections[0].I_skin*1e12:.2f} mm4")
    print(f"    Skin I fraction = {box_sections[0].skin_fraction*100:.1f}%")

    print(f"\n  Critical station (idx={max_vm_idx}, y={y[max_vm_idx]*1000:.1f}mm):")
    print(f"    sigma_b_FS = {crit_box.sigma_b_FS/1e6:.4f} MPa")
    print(f"    tau_FS = {crit_box.tau_FS/1e6:.4f} MPa")
    print(f"    sigma_vm_FS = {crit_box.sigma_vm_FS/1e6:.4f} MPa")
    print(f"    sigma_b_RS = {crit_box.sigma_b_RS/1e6:.4f} MPa")
    print(f"    tau_RS = {crit_box.tau_RS/1e6:.4f} MPa")
    print(f"    sigma_vm_RS = {crit_box.sigma_vm_RS/1e6:.4f} MPa")
    print(f"    sigma_skin_comp_max = {max_skin_comp/1e6:.6f} MPa")

    # Interaction checks
    FS_interaction = (crit_box.sigma_b_FS / sigma_allow_spar) + (crit_box.tau_FS / tau_allow_spar) ** 2
    RS_interaction = (crit_box.sigma_b_RS / sigma_allow_spar) + (crit_box.tau_RS / tau_allow_spar) ** 2

    print(f"\n  Spar interaction:")
    print(f"    FS: ({crit_box.sigma_b_FS/1e6:.4f}/{sigma_allow_spar/1e6:.1f}) + ({crit_box.tau_FS/1e6:.4f}/{tau_allow_spar/1e6:.1f})^2 = {FS_interaction:.6f}")
    print(f"    RS: ({crit_box.sigma_b_RS/1e6:.4f}/{sigma_allow_spar/1e6:.1f}) + ({crit_box.tau_RS/1e6:.4f}/{tau_allow_spar/1e6:.1f})^2 = {RS_interaction:.6f}")

    p2_pass = True

    fs_int_check = check_result("FS interaction", FS_interaction, 1.0, "<")
    if not fs_int_check:
        p2_pass = False

    rs_int_check = check_result("RS interaction", RS_interaction, 1.0, "<")
    if not rs_int_check:
        p2_pass = False

    # Critical areas
    sigma_max_spar = mat_spar.sigma_u
    c_FS_dist = spar_FS.d_outer / 2
    c_RS_dist = spar_RS.d_outer / 2

    A_Cri_FS = (n * W_0 * planform.Y_bar * 0.5 * sharing.eta_FS) / (2 * sigma_max_spar * c_FS_dist)
    A_Cri_RS = (n * W_0 * planform.Y_bar * 0.5 * sharing.eta_RS) / (2 * sigma_max_spar * c_RS_dist)

    print(f"\n  Area checks:")
    print(f"    A_Act_FS = {spar_FS.area*1e6:.2f} mm2, A_Cri_FS = {A_Cri_FS*1e6:.2f} mm2")
    print(f"    A_Act_RS = {spar_RS.area*1e6:.2f} mm2, A_Cri_RS = {A_Cri_RS*1e6:.2f} mm2")

    fs_area_check = check_result("FS area", spar_FS.area*1e6, A_Cri_FS*1e6, ">", "mm2")
    if not fs_area_check:
        p2_pass = False

    rs_area_check = check_result("RS area", spar_RS.area*1e6, A_Cri_RS*1e6, ">", "mm2")
    if not rs_area_check:
        p2_pass = False

    print(f"\n  PHASE 2 RESULT: {'PASS' if p2_pass else '*** FAIL ***'}")

    # =========================================================================
    # STEP 5: Phase 3 - Buckling (NEW - not in old version)
    # =========================================================================
    banner("PHASE 3: Adaptive Rib Insertion + Buckling")

    print(f"  This phase was ADDED in the new version!")
    print(f"  Old version: only Phase 0-2")
    print(f"  New version: Phase 0-2 + Phase 3 (buckling)")

    # Calculate rib spacing
    rib_spacing = L_span / N_Rib
    print(f"\n  Rib spacing = {rib_spacing*1000:.2f} mm (N_Rib={N_Rib})")
    print(f"  s_min = {s_min_mm} mm")

    # Buckling critical stresses for default spacing
    E_pla = materials.skin.E
    nu_pla = materials.skin.nu

    print(f"\n  PLA skin properties for buckling:")
    print(f"    E = {E_pla/1e9:.1f} GPa")
    print(f"    nu = {nu_pla}")
    print(f"    t_skin = {t_skin*1000:.3f} mm")

    # Show how tau_cr and sigma_cr vary with spacing
    print(f"\n  Buckling critical stress vs spacing:")
    print(f"  {'s [mm]':>8} {'tau_cr [MPa]':>14} {'sigma_cr [MPa]':>16}")
    for s_mm in [20, 50, 100, 150, 200, 300, rib_spacing*1000]:
        s = s_mm / 1000
        tau_cr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, s)
        sigma_cr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, s)
        marker = " <-- default spacing" if abs(s_mm - rib_spacing*1000) < 0.1 else ""
        print(f"  {s_mm:8.1f} {tau_cr/1e6:14.6f} {sigma_cr/1e6:16.6f}{marker}")

    # Actual applied stresses
    tau_skin_arr = np.array([abs(tr.tau_skin) for tr in torsion_results])
    sigma_comp_arr = np.array([br.sigma_skin_comp for br in box_results])

    print(f"\n  Applied stresses at root (worst case):")
    print(f"    tau_skin_root = {tau_skin_arr[0]/1e6:.6f} MPa")
    print(f"    sigma_comp_root = {sigma_comp_arr[0]/1e6:.6f} MPa")

    tau_cr_default = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, rib_spacing)
    sigma_cr_default = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, rib_spacing)

    print(f"\n  Buckling criticals at default spacing ({rib_spacing*1000:.2f} mm):")
    print(f"    tau_cr = {tau_cr_default/1e6:.6f} MPa")
    print(f"    sigma_cr = {sigma_cr_default/1e6:.6f} MPa")

    # Compare
    print(f"\n  Root bay comparison:")
    print(f"    tau_skin / tau_cr = {tau_skin_arr[0]/tau_cr_default:.4f} (must be < 1)")
    print(f"    sigma_comp / sigma_cr = {sigma_comp_arr[0]/sigma_cr_default:.4f} (must be < 1)")

    R_root = combined_interaction_ratio(tau_skin_arr[0], tau_cr_default,
                                        sigma_comp_arr[0], sigma_cr_default)
    print(f"    R_combined = {R_root:.6f} (must be < 1)")

    # What spacing would be needed?
    s_max_shear = compute_s_max_shear(E_pla, nu_pla, t_skin, tau_skin_arr[0])
    s_max_comp = compute_s_max_compression(E_pla, nu_pla, t_skin, sigma_comp_arr[0])

    print(f"\n  Maximum allowable spacing for buckling:")
    print(f"    s_max_shear = {s_max_shear*1000:.2f} mm")
    print(f"    s_max_comp  = {s_max_comp*1000:.2f} mm")
    print(f"    s_governing = {min(s_max_shear, s_max_comp)*1000:.2f} mm")
    print(f"    s_governing * 0.95 = {min(s_max_shear, s_max_comp)*0.95*1000:.2f} mm")
    print(f"    Actual spacing = {rib_spacing*1000:.2f} mm")

    if rib_spacing > min(s_max_shear, s_max_comp):
        print(f"    -> Spacing EXCEEDS buckling limit! Need more ribs.")
        required_N = int(np.ceil(L_span / min(s_max_shear, s_max_comp)))
        print(f"    -> Minimum N_Rib for buckling = {required_N}")
    else:
        print(f"    -> Spacing OK for buckling.")

    # Adaptive rib insertion
    print(f"\n  Running adaptive rib insertion...")

    tau_interp = interp1d(y, tau_skin_arr, kind='linear', fill_value='extrapolate')
    sigma_interp = interp1d(y, sigma_comp_arr, kind='linear', fill_value='extrapolate')
    V_interp = interp1d(y, loads.V, kind='linear', fill_value='extrapolate')
    h_interp = interp1d(y, h_box, kind='linear', fill_value='extrapolate')

    adapt_config = AdaptiveRibConfig(
        E_skin=E_pla,
        nu_skin=nu_pla,
        t_skin=t_skin,
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
    print(f"  Result: {adapt_msg}")
    print(f"  N_Rib_initial = {N_Rib}")
    print(f"  N_Rib_final = {N_Rib_final}")
    print(f"  Feasible = {feasible}")

    if not feasible:
        print(f"\n  *** INFEASIBLE: Adaptive rib insertion hit s_min = {s_min_mm} mm")
        print(f"  This means even with maximum rib density, buckling still fails.")

        # Check: what spacing would the most stressed bay need?
        min_spacing = final_y_ribs[1] - final_y_ribs[0]
        for i in range(len(final_y_ribs) - 1):
            sp = final_y_ribs[i+1] - final_y_ribs[i]
            if sp < min_spacing:
                min_spacing = sp
        print(f"  Minimum bay spacing achieved: {min_spacing*1000:.2f} mm")

    # Check each bay after adaptive insertion
    print(f"\n  Bay-by-bay buckling check after adaptive insertion:")
    print(f"  {'Bay':>4} {'y_s[mm]':>8} {'y_e[mm]':>8} {'s[mm]':>7} {'tau':>10} {'tau_cr':>10} {'sig':>10} {'sig_cr':>10} {'R':>8} {'Status'}")

    p3_pass = True
    tau_cr_min = float('inf')
    sigma_cr_min = float('inf')
    R_max = 0

    for i in range(len(final_y_ribs) - 1):
        y_s = final_y_ribs[i]
        y_e = final_y_ribs[i + 1]
        spacing = y_e - y_s

        tau_p = max(abs(tau_interp(y_s)), abs(tau_interp(y_e)))
        sig_p = max(abs(sigma_interp(y_s)), abs(sigma_interp(y_e)))

        tau_cr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, spacing, 5.34)
        sigma_cr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, spacing, 4.0)
        R = combined_interaction_ratio(tau_p, tau_cr, sig_p, sigma_cr)

        if tau_cr < tau_cr_min:
            tau_cr_min = tau_cr
        if sigma_cr < sigma_cr_min:
            sigma_cr_min = sigma_cr
        if R > R_max:
            R_max = R

        bay_pass = (tau_p <= tau_cr) and (sig_p <= sigma_cr) and (R < 1.0)
        status = "PASS" if bay_pass else "FAIL"
        if not bay_pass:
            p3_pass = False

        # Print first 10 + last 3 bays, or all if <= 20
        if N_Rib_final <= 20 or i < 10 or i >= N_Rib_final - 3:
            print(f"  {i+1:4d} {y_s*1000:8.1f} {y_e*1000:8.1f} {spacing*1000:7.1f} "
                  f"{tau_p/1e6:10.4f} {tau_cr/1e6:10.4f} "
                  f"{sig_p/1e6:10.4f} {sigma_cr/1e6:10.4f} "
                  f"{R:8.4f} {status}")
        elif i == 10:
            print(f"  ... ({N_Rib_final - 13} bays omitted) ...")

    print(f"\n  Buckling summary:")
    print(f"    tau_cr_min = {tau_cr_min/1e6:.6f} MPa")
    print(f"    sigma_cr_min = {sigma_cr_min/1e6:.6f} MPa")
    print(f"    sigma_skin_comp_max = {max_skin_comp/1e6:.6f} MPa")
    print(f"    R_combined_max = {R_max:.6f}")

    # Phase 3 specific checks
    print(f"\n  Phase 3 checks:")

    # 8.9 Skin compression
    if sigma_cr_min > 0:
        comp_buck_check = check_result("sigma_comp_max vs sigma_cr_min",
                                        max_skin_comp/1e6, sigma_cr_min/1e6, "<", "MPa")
        if not comp_buck_check:
            p3_pass = False

    # 8.10 Combined interaction
    combined_check = check_result("R_combined_max", R_max, 1.0, "<")
    if not combined_check:
        p3_pass = False

    # Feasibility
    if not feasible:
        print(f"  Adaptive insertion: *** FAIL (infeasible spacing) ***")
        p3_pass = False
    else:
        print(f"  Adaptive insertion: PASS (feasible)")

    print(f"\n  PHASE 3 RESULT: {'PASS' if p3_pass else '*** FAIL ***'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    banner("SUMMARY: Why No Solutions?")

    all_pass = p0_pass and p1_pass and p2_pass and p3_pass

    print(f"  Phase 0 (Geometry/Assembly): {'PASS' if p0_pass else 'FAIL'}")
    print(f"  Phase 1 (Skin Stress/Twist): {'PASS' if p1_pass else 'FAIL'}")
    print(f"  Phase 2 (Spar Interaction/Area): {'PASS' if p2_pass else 'FAIL'}")
    print(f"  Phase 3 (Buckling): {'PASS' if p3_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        print(f"\n  Root cause analysis:")
        if not p0_pass:
            print(f"    - Geometry/assembly constraint violated")
            print(f"      d_spar + 10 must be < h_box_root = {h_box_root_mm:.1f} mm")
        if not p1_pass:
            if not tau_check:
                print(f"    - Skin shear stress exceeded ({tau_skin_max/1e6:.4f} > {tau_allow_skin/1e6:.1f} MPa)")
            if not twist_check:
                print(f"    - Tip twist exceeded ({theta_tip_deg:.2f} > {theta_max_deg:.1f} deg)")
        if not p2_pass:
            if not fs_int_check:
                print(f"    - FS spar interaction formula exceeded")
            if not rs_int_check:
                print(f"    - RS spar interaction formula exceeded")
            if not fs_area_check:
                print(f"    - FS actual area less than critical area")
            if not rs_area_check:
                print(f"    - RS actual area less than critical area")
        if not p3_pass:
            print(f"    - BUCKLING is the killer!")
            print(f"      With PLA skin (E={E_pla/1e9:.1f} GPa) and t_skin={t_skin_mm} mm,")
            print(f"      the buckling critical stresses are VERY LOW.")
            print(f"      For spacing {rib_spacing*1000:.0f} mm: tau_cr = {tau_cr_default/1e6:.4f} MPa")
            print(f"      Applied tau_skin at root = {tau_skin_arr[0]/1e6:.4f} MPa")
            if not feasible:
                print(f"      Even at s_min = {s_min_mm} mm, buckling cannot be satisfied!")
                s_min_tau_cr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, s_min_mm/1000)
                s_min_sigma_cr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, s_min_mm/1000)
                print(f"      At s = {s_min_mm} mm: tau_cr = {s_min_tau_cr/1e6:.4f} MPa, sigma_cr = {s_min_sigma_cr/1e6:.4f} MPa")

    # =========================================================================
    # GRID SEARCH SCAN: How many configs pass Phase 0-2 but fail Phase 3?
    # =========================================================================
    banner("GRID SEARCH SCAN: Phase 0-2 pass rates")

    print(f"  Scanning grid search ranges to find what passes Phase 0-2...")

    # Grid ranges (default)
    t_rib_range = np.arange(1.0, 5.5, 1.0)
    X_FS_range = np.arange(15.0, 32.5, 5.0)
    X_RS_range = np.arange(40.0, 67.5, 5.0)
    d_FS_range = np.arange(16.0, 31.0, 2.0)
    t_FS_range = np.arange(0.5, 3.25, 0.5)
    d_RS_range = np.arange(10.0, 27.5, 5.0)
    t_RS_range = np.arange(0.5, 3.25, 0.5)

    total = (len(t_rib_range) * len(X_FS_range) * len(X_RS_range) *
             len(d_FS_range) * len(t_FS_range) * len(d_RS_range) * len(t_RS_range))

    print(f"  Total combinations: {total}")

    count = 0
    p0_count = 0
    p1_count = 0
    p2_count = 0
    p3_count = 0

    # Track failure reasons
    fail_reasons = {}

    # Track some configs that pass Phase 0-2
    p2_passed_configs = []

    for t_rib_v in t_rib_range:
        for X_FS_v in X_FS_range:
            for X_RS_v in X_RS_range:
                for d_FS_v in d_FS_range:
                    for t_FS_v in t_FS_range:
                        for d_RS_v in d_RS_range:
                            for t_RS_v in t_RS_range:
                                count += 1

                                # Phase 0 checks
                                if X_FS_v >= X_RS_v:
                                    reason = "X_FS >= X_RS"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue
                                if t_FS_v >= d_FS_v / 2:
                                    reason = "t_FS >= d_FS/2"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue
                                if t_RS_v >= d_RS_v / 2:
                                    reason = "t_RS >= d_RS/2"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue

                                # Assembly
                                if d_FS_v + 10 >= h_box_root_mm:
                                    reason = "FS assembly"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue
                                if d_RS_v + 10 >= h_box_root_mm:
                                    reason = "RS assembly"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue

                                p0_count += 1

                                # Phase 1: we already computed loads/torsion for the default
                                # But Phase 1 depends on X_FS, X_RS (spar positions change geometry)
                                # For a QUICK scan, let's assume Phase 1 passes if default passes
                                # (since tau_skin is dominated by T/Am and T/Am changes slowly)

                                # Phase 1 pass (skin stress and twist) - approximate
                                # The torsion depends on spar positions via shear center
                                # For a scan, let's just count Phase 0 pass and check Phase 2

                                # Phase 2 quick check: spar interaction
                                spar_FS_check = SparProperties.from_mm(d_FS_v, t_FS_v, mat_spar)
                                spar_RS_check = SparProperties.from_mm(d_RS_v, t_RS_v, mat_spar)

                                sharing_check = LoadSharing.from_positions(X_FS_v, X_RS_v, x_ac_percent)

                                # Use root loads (worst case)
                                M_root = loads.M[0]
                                T_root = loads.T[0]

                                # Quick box beam at root
                                h_root = h_box[0]
                                w_root = box_width[0]  # Approximate - changes with spar pos

                                box_root = compute_box_beam_section(spar_FS_check, spar_RS_check,
                                                                     h_root, w_root, t_skin, E_skin)
                                stress_root = analyze_box_beam_stress(M_root, T_root, box_root,
                                                                      spar_FS_check, spar_RS_check,
                                                                      sharing_check.eta_FS, sharing_check.eta_RS)

                                # Check interaction
                                FS_int = (stress_root.sigma_b_FS / sigma_allow_spar) + (stress_root.tau_FS / tau_allow_spar)**2
                                RS_int = (stress_root.sigma_b_RS / sigma_allow_spar) + (stress_root.tau_RS / tau_allow_spar)**2

                                if FS_int >= 1.0:
                                    reason = "FS interaction"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue
                                if RS_int >= 1.0:
                                    reason = "RS interaction"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue

                                # Area checks
                                c_FS_d = spar_FS_check.d_outer / 2
                                c_RS_d = spar_RS_check.d_outer / 2
                                A_Cri_FS_c = (n * W_0 * planform.Y_bar * 0.5 * sharing_check.eta_FS) / (2 * sigma_max_spar * c_FS_d)
                                A_Cri_RS_c = (n * W_0 * planform.Y_bar * 0.5 * sharing_check.eta_RS) / (2 * sigma_max_spar * c_RS_d)

                                if spar_FS_check.area <= A_Cri_FS_c:
                                    reason = "FS area"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue
                                if spar_RS_check.area <= A_Cri_RS_c:
                                    reason = "RS area"
                                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                                    continue

                                p2_count += 1

                                if len(p2_passed_configs) < 5:
                                    p2_passed_configs.append({
                                        'X_FS': X_FS_v, 'X_RS': X_RS_v,
                                        'd_FS': d_FS_v, 't_FS': t_FS_v,
                                        'd_RS': d_RS_v, 't_RS': t_RS_v,
                                        't_rib': t_rib_v,
                                        'FS_int': FS_int, 'RS_int': RS_int,
                                    })

    print(f"\n  Results:")
    print(f"    Total configs: {total}")
    print(f"    Pass Phase 0 (geometry/assembly): {p0_count} ({p0_count/total*100:.1f}%)")
    print(f"    Pass Phase 0+2 (+ spar checks): {p2_count} ({p2_count/total*100:.1f}%)")

    print(f"\n  Failure breakdown:")
    for reason, cnt in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"    {reason}: {cnt} ({pct:.1f}%)")

    # Show some configs that pass Phase 0-2
    if p2_passed_configs:
        print(f"\n  Sample configs that PASS Phase 0-2 (these would have passed in old version):")
        for i, cfg in enumerate(p2_passed_configs):
            print(f"    [{i+1}] X_FS={cfg['X_FS']}%, X_RS={cfg['X_RS']}%, "
                  f"d_FS={cfg['d_FS']}mm, t_FS={cfg['t_FS']}mm, "
                  f"d_RS={cfg['d_RS']}mm, t_RS={cfg['t_RS']}mm, "
                  f"FS_int={cfg['FS_int']:.4f}, RS_int={cfg['RS_int']:.4f}")

    # =========================================================================
    # Now check Phase 3 for the configs that pass Phase 0-2
    # =========================================================================
    if p2_passed_configs:
        banner("Phase 3 CHECK on Phase 0-2 passed configs")

        for i, cfg in enumerate(p2_passed_configs[:3]):
            print(f"\n  Config [{i+1}]: X_FS={cfg['X_FS']}%, X_RS={cfg['X_RS']}%, "
                  f"d_FS={cfg['d_FS']}mm, d_RS={cfg['d_RS']}mm")

            # Re-do full analysis for this config
            sp = SparPosition(X_FS_percent=cfg['X_FS'], X_RS_percent=cfg['X_RS'])
            sts = compute_all_stations(planform, sp, N_Rib)

            y_c = np.array([s.y for s in sts])
            chord_c = np.array([s.chord for s in sts])
            xFS_c = np.array([s.x_FS for s in sts])
            xRS_c = np.array([s.x_RS for s in sts])
            Am_c = np.array([s.A_m for s in sts])
            hbox_c = np.array([s.h_box for s in sts])
            bw_c = np.array([s.width for s in sts])

            loads_c = analyze_loads(y_c, chord_c, xFS_c, xRS_c, flight, aero_center,
                                    planform.L_span,
                                    LoadDistributionType.UNIFORM,
                                    PitchMomentDistributionType.UNIFORM)

            torsion_c = analyze_torsion_stations(y_c, loads_c.T, Am_c, t_skin, G_skin, bw_c, hbox_c)

            sp_FS_c = SparProperties.from_mm(cfg['d_FS'], cfg['t_FS'], mat_spar)
            sp_RS_c = SparProperties.from_mm(cfg['d_RS'], cfg['t_RS'], mat_spar)

            box_c = [
                compute_box_beam_section(sp_FS_c, sp_RS_c, hbox_c[j], bw_c[j], t_skin, E_skin)
                for j in range(len(y_c))
            ]

            sharing_c = LoadSharing.from_positions(cfg['X_FS'], cfg['X_RS'], x_ac_percent)

            box_res_c = [
                analyze_box_beam_stress(loads_c.M[j], loads_c.T[j], box_c[j],
                                        sp_FS_c, sp_RS_c, sharing_c.eta_FS, sharing_c.eta_RS, y_c[j])
                for j in range(len(y_c))
            ]

            tau_arr_c = np.array([abs(tr.tau_skin) for tr in torsion_c])
            sig_arr_c = np.array([br.sigma_skin_comp for br in box_res_c])

            tau_i = interp1d(y_c, tau_arr_c, kind='linear', fill_value='extrapolate')
            sig_i = interp1d(y_c, sig_arr_c, kind='linear', fill_value='extrapolate')
            V_i = interp1d(y_c, loads_c.V, kind='linear', fill_value='extrapolate')
            h_i = interp1d(y_c, hbox_c, kind='linear', fill_value='extrapolate')

            ac = AdaptiveRibConfig(
                E_skin=E_pla, nu_skin=nu_pla, t_skin=t_skin,
                E_rib=materials.rib.E, nu_rib=materials.rib.nu,
                t_rib=cfg['t_rib']/1000,
                tau_allow_skin=tau_allow_skin, sigma_allow_skin=sigma_allow_skin,
                k_s=5.34, k_c=4.0, s_min=s_min_mm/1000, margin=0.95,
            )

            final_y, feas, msg = adaptive_rib_insertion(y_c, ac, tau_i, sig_i, V_i, h_i)

            print(f"    Adaptive insertion: feasible={feas}, N_Rib_final={len(final_y)-1}")

            # Check root bay
            if len(final_y) > 1:
                s0 = final_y[1] - final_y[0]
                tau_root = abs(tau_i(final_y[0]))
                sig_root = abs(sig_i(final_y[0]))
                tcr0 = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, s0)
                scr0 = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, s0)
                R0 = combined_interaction_ratio(tau_root, tcr0, sig_root, scr0)

                print(f"    Root bay: s={s0*1000:.1f}mm, tau={tau_root/1e6:.4f} MPa, tau_cr={tcr0/1e6:.4f} MPa")
                print(f"    Root bay: sig={sig_root/1e6:.6f} MPa, sig_cr={scr0/1e6:.4f} MPa")
                print(f"    Root bay: R={R0:.4f} ({'PASS' if R0 < 1.0 else 'FAIL'})")

            if not feas:
                # What t_skin would be needed?
                for ts_try in [1.0, 1.5, 2.0, 3.0]:
                    ts_m = ts_try / 1000
                    tcr_try = skin_shear_buckling_critical(E_pla, nu_pla, ts_m, s_min_mm/1000)
                    scr_try = skin_compression_buckling_critical(E_pla, nu_pla, ts_m, s_min_mm/1000)
                    print(f"    IF t_skin={ts_try:.1f}mm, s={s_min_mm}mm: tau_cr={tcr_try/1e6:.3f}, sig_cr={scr_try/1e6:.3f} MPa")

    # =========================================================================
    # WHAT WOULD FIX IT?
    # =========================================================================
    banner("POTENTIAL FIXES")

    print(f"  The core problem: PLA skin is too flexible for buckling resistance.")
    print(f"  E_PLA = {E_pla/1e9:.1f} GPa vs E_AL = 71.7 GPa vs E_CFRP = 135 GPa")
    print(f"")
    print(f"  Option 1: Increase t_skin")
    print(f"    tau_cr ~ (t/s)^2, so doubling t_skin -> 4x tau_cr")
    for ts in [0.625, 1.0, 1.5, 2.0, 3.0]:
        ts_m = ts / 1000
        tcr = skin_shear_buckling_critical(E_pla, nu_pla, ts_m, rib_spacing)
        scr = skin_compression_buckling_critical(E_pla, nu_pla, ts_m, rib_spacing)
        R_test = combined_interaction_ratio(tau_skin_arr[0], tcr, sigma_comp_arr[0], scr)
        print(f"    t_skin={ts:.3f}mm: tau_cr={tcr/1e6:.4f}, sig_cr={scr/1e6:.4f}, R={R_test:.4f} "
              f"{'<- default' if ts == 0.625 else ''}")

    print(f"\n  Option 2: Different skin material (e.g., AL7075)")
    al_mat = db.get_material('AL7075-T6')
    for ts in [0.625, 1.0, 1.5]:
        ts_m = ts / 1000
        tcr = skin_shear_buckling_critical(al_mat.E, al_mat.nu, ts_m, rib_spacing)
        scr = skin_compression_buckling_critical(al_mat.E, al_mat.nu, ts_m, rib_spacing)
        R_test = combined_interaction_ratio(tau_skin_arr[0], tcr, sigma_comp_arr[0], scr)
        print(f"    AL7075, t_skin={ts:.3f}mm: tau_cr={tcr/1e6:.4f}, sig_cr={scr/1e6:.4f}, R={R_test:.4f}")

    print(f"\n  Option 3: Relax s_min constraint (allow denser ribs)")
    for sm in [20, 15, 10, 5]:
        sm_m = sm / 1000
        tcr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, sm_m)
        scr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, sm_m)
        R_test = combined_interaction_ratio(tau_skin_arr[0], tcr, sigma_comp_arr[0], scr)
        print(f"    s_min={sm}mm: tau_cr={tcr/1e6:.4f}, sig_cr={scr/1e6:.4f}, R={R_test:.4f} "
              f"{'<- PASS' if R_test < 1.0 else ''}")

    print(f"\n  Option 4: Disable Phase 3 / buckling (like old version)")
    print(f"    This would restore the old behavior with {p2_count} solutions")
    print(f"    BUT those solutions would be structurally unsound (buckling failure)")

    print(f"\n  Option 5: Increase t_skin AND reduce s_min")
    for ts, sm in [(1.0, 20), (1.0, 15), (1.5, 20), (1.5, 15), (2.0, 20)]:
        ts_m = ts / 1000
        sm_m = sm / 1000
        tcr = skin_shear_buckling_critical(E_pla, nu_pla, ts_m, sm_m)
        scr = skin_compression_buckling_critical(E_pla, nu_pla, ts_m, sm_m)
        R_test = combined_interaction_ratio(tau_skin_arr[0], tcr, sigma_comp_arr[0], scr)
        print(f"    t_skin={ts:.1f}mm, s_min={sm}mm: tau_cr={tcr/1e6:.4f}, sig_cr={scr/1e6:.4f}, R={R_test:.4f} "
              f"{'<- PASS' if R_test < 1.0 else ''}")


if __name__ == "__main__":
    main()

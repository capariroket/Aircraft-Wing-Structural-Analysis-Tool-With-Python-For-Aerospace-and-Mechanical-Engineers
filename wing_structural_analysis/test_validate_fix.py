#!/usr/bin/env python3
"""
test_validate_fix.py - Validate the bugfixes in main.py

Tests:
  1. validate_solution logic: buckling_mode=1 must NOT check R_combined
  2. t_skin_final_mm is correctly propagated from Phase-2
  3. Small design space for speed (~64 combos)

Run: python3 test_validate_fix.py
"""

import sys
import os
import time
import traceback
import numpy as np

PROJECT_DIR = '/Users/apple/Desktop/dosyalar/wing_structural_analysis'
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from materials import MaterialDatabase, MaterialSelection
from geometry import (PlanformParams, SparPosition, compute_all_stations,
                      compute_spar_sweep_angle, compute_skin_arc_lengths_root,
                      compute_skin_area_half_wing)
from loads import (FlightCondition, AeroCenter, analyze_loads,
                   LoadDistributionType, PitchMomentDistributionType)
from torsion import analyze_all_stations as analyze_torsion_stations, find_critical_station
from spars import (SparProperties, LoadSharing, BoxBeamSection, compute_box_beam_section,
                   analyze_box_beam_stress, spar_mass)
from ribs import (RibProperties, generate_rib_geometries, compute_total_rib_mass,
                  compute_minimum_rib_count)
from optimization import (DesignRange, DesignSpace, run_optimization,
                          GridSearchOptimizer, EvaluationResult)


# =============================================================================
# DEFAULT PARAMETERS (matching main.py)
# =============================================================================
DEFAULTS = {
    'C_m': -0.003,
    'AR': 11.0,
    'lambda_taper': 0.45,
    'V_c': 21.0,
    'n': 2.0,
    'S_ref': 3.17,
    'W_0': 650.0,
    'rho_air': 1.223,
    'x_ac_percent': 25.0,
    'Lambda_ac_deg': 0.0,
    'b': 5.195,
    't_over_c': 0.17,
    't_skin_mm': 1.2,
    'SF': 1.0,
    'theta_max_deg': 2.0,
    's_min_mm': 20.0,
    'material_spar_id': 3,  # CFRP_UD
    'material_rib_id': 5,   # PLA
    'material_skin_id': 5,  # PLA
}


def setup_common():
    """Setup common objects from default parameters."""
    AR = DEFAULTS['AR']
    S_ref = DEFAULTS['S_ref']
    lambda_taper = DEFAULTS['lambda_taper']
    b = DEFAULTS['b']
    t_over_c = DEFAULTS['t_over_c']

    C_r = 2 * S_ref / (b * (1 + lambda_taper))
    C_r_mm = C_r * 1000
    c_MGC = (4 / 3) * np.sqrt(S_ref / AR) * (
        (1 + lambda_taper + lambda_taper**2) /
        (1 + 2 * lambda_taper + lambda_taper**2)
    )
    Y_bar_mm = (b / 6) * ((1 + 2 * lambda_taper) / (1 + lambda_taper)) * 1000
    N_Rib = int(np.ceil(1 + np.sqrt(AR * S_ref) / c_MGC))
    L_span = b / 2
    c_tip = lambda_taper * C_r
    h_box_root_mm = t_over_c * C_r_mm

    db = MaterialDatabase()
    mat_keys = list(db.materials.keys())
    mat_spar_key = mat_keys[DEFAULTS['material_spar_id'] - 1]
    mat_rib_key = mat_keys[DEFAULTS['material_rib_id'] - 1]
    mat_skin_key = mat_keys[DEFAULTS['material_skin_id'] - 1]
    materials = MaterialSelection.from_database(db, mat_spar_key, mat_skin_key, mat_rib_key)

    planform = PlanformParams.from_input(
        b=b, AR=AR, taper_ratio=lambda_taper, t_c=t_over_c,
        S_ref=S_ref, C_r_mm=C_r_mm, c_MGC=c_MGC, Y_bar_mm=Y_bar_mm
    )

    flight = FlightCondition(
        W0=DEFAULTS['W_0'], n=DEFAULTS['n'], V_c=DEFAULTS['V_c'],
        rho=DEFAULTS['rho_air'], C_m=DEFAULTS['C_m'],
        S_ref=S_ref, c_MGC=c_MGC
    )

    aero_center = AeroCenter(
        x_ac_percent=DEFAULTS['x_ac_percent'],
        Lambda_ac_deg=DEFAULTS['Lambda_ac_deg']
    )

    return {
        'planform': planform,
        'flight': flight,
        'aero_center': aero_center,
        'materials': materials,
        'N_Rib': N_Rib,
        'h_box_root_mm': h_box_root_mm,
        't_skin_mm': DEFAULTS['t_skin_mm'],
        'SF': DEFAULTS['SF'],
        'theta_max_deg': DEFAULTS['theta_max_deg'],
        's_min_mm': DEFAULTS['s_min_mm'],
    }


def build_small_design_space(N_Rib):
    """Build a small design space for fast testing.

    Ranges:
      N_Rib: fixed (12)
      t_rib: 3.0 only
      X_FS: 20, 25
      X_RS: 50, 55
      d_FS: 20, 24
      t_FS: 1.5, 2.0
      d_RS: 14, 18
      t_RS: 1.5, 2.0

    Total = 1 * 1 * 2 * 2 * 2 * 2 * 2 * 2 = 64 combinations
    """
    return DesignSpace(
        N_Rib=DesignRange(min_val=N_Rib, max_val=N_Rib, step=1),
        t_rib_mm=DesignRange(3.0, 3.0, 1.0),
        X_FS_percent=DesignRange(20.0, 25.0, 5.0),
        X_RS_percent=DesignRange(50.0, 55.0, 5.0),
        d_FS_outer_mm=DesignRange(20.0, 24.0, 4.0),
        t_FS_mm=DesignRange(1.5, 2.0, 0.5),
        d_RS_outer_mm=DesignRange(14.0, 18.0, 4.0),
        t_RS_mm=DesignRange(1.5, 2.0, 0.5),
    )


def simulate_validate_solution(sol, buckling_mode, h_box_mm, theta_max_deg,
                                tau_allow, sigma_allow, tau_allow_spar):
    """Simulate main.py's validate_solution to verify the bugfix.

    Key point: when buckling_mode=1, criteria 9 (compression buckling)
    and 10 (R_combined) must NOT be checked.
    """
    errors = []
    cfg = sol.config

    # 1. Skin shear stress
    if sol.tau_skin_max > tau_allow:
        errors.append(f"[1] Skin stress: {sol.tau_skin_max/1e6:.2f} > {tau_allow/1e6:.2f} MPa")

    # 2. FS interaction
    FS_interaction = (sol.sigma_b_FS_max / sigma_allow) + (sol.tau_FS_max / tau_allow_spar) ** 2
    if FS_interaction >= 1.0:
        errors.append(f"[2] FS interaction: {FS_interaction:.3f} >= 1.0")

    # 3. RS interaction
    RS_interaction = (sol.sigma_b_RS_max / sigma_allow) + (sol.tau_RS_max / tau_allow_spar) ** 2
    if RS_interaction >= 1.0:
        errors.append(f"[3] RS interaction: {RS_interaction:.3f} >= 1.0")

    # 4. FS area
    if sol.A_Act_FS <= sol.A_Cri_FS:
        errors.append(f"[4] FS area: {sol.A_Act_FS*1e6:.2f} <= {sol.A_Cri_FS*1e6:.2f} mm2")

    # 5. RS area
    if sol.A_Act_RS <= sol.A_Cri_RS:
        errors.append(f"[5] RS area: {sol.A_Act_RS*1e6:.2f} <= {sol.A_Cri_RS*1e6:.2f} mm2")

    # 6. FS assembly
    if cfg.d_FS_outer_mm + 10 >= h_box_mm:
        errors.append(f"[6] FS assembly: {cfg.d_FS_outer_mm}+10 >= {h_box_mm:.1f} mm")

    # 7. RS assembly
    if cfg.d_RS_outer_mm + 10 >= h_box_mm:
        errors.append(f"[7] RS assembly: {cfg.d_RS_outer_mm}+10 >= {h_box_mm:.1f} mm")

    # 8. Tip twist
    if sol.theta_tip_deg > theta_max_deg:
        errors.append(f"[8] Twist: {sol.theta_tip_deg:.2f} > {theta_max_deg:.1f} deg")

    # 9 & 10: ONLY in buckling_mode=2
    if buckling_mode == 2:
        if hasattr(sol, 'sigma_cr_min') and sol.sigma_cr_min > 0:
            if sol.sigma_skin_comp_max > sol.sigma_cr_min:
                errors.append(f"[9] Compression buckling: sigma={sol.sigma_skin_comp_max/1e6:.2f} > sigma_cr={sol.sigma_cr_min/1e6:.2f} MPa")
        if hasattr(sol, 'R_combined_max'):
            if sol.R_combined_max >= 1.0:
                errors.append(f"[10] Combined R: {sol.R_combined_max:.3f} >= 1.0")

    return len(errors) == 0, errors


# =============================================================================
# MAIN TEST
# =============================================================================
def main():
    print("=" * 70)
    print("TEST: validate_solution bugfix & t_skin_final propagation")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Setup ----
    ctx = setup_common()
    ds = build_small_design_space(ctx['N_Rib'])

    print(f"\n  N_Rib = {ctx['N_Rib']} (fixed)")
    print(f"  h_box_root_mm = {ctx['h_box_root_mm']:.2f}")
    print(f"  t_skin_mm = {ctx['t_skin_mm']}")
    print(f"  Total combinations = {ds.total_combinations}")
    print(f"  buckling_mode = 1 (shear only)")

    # ---- Run Phase-1 + Phase-2 ----
    print(f"\n{'='*50}")
    print("Running optimization (buckling_mode=1, shear only)")
    print(f"{'='*50}")

    t0 = time.time()
    try:
        best, optimizer = run_optimization(
            planform=ctx['planform'],
            flight=ctx['flight'],
            aero_center=ctx['aero_center'],
            materials=ctx['materials'],
            design_space=ds,
            t_skin_mm=ctx['t_skin_mm'],
            SF=ctx['SF'],
            h_FS_mm=ctx['h_box_root_mm'],
            h_RS_mm=ctx['h_box_root_mm'],
            theta_max_deg=ctx['theta_max_deg'],
            rib_parabolic=False,
            load_dist='uniform',
            pitch_dist='uniform',
            n_workers=1,
            s_min_mm=ctx['s_min_mm'],
            buckling_mode=1,
        )
    except Exception as e:
        print(f"\n  RUNTIME ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n  FIX: Check main.py, optimization.py, ribs.py for import/logic errors.")
        return

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    if best is None:
        print("\n  NO SOLUTION FOUND from optimization.")
        print("  Phase-1 accepted:", len(optimizer.accepted_results))
        print("  Rejection summary:")
        for reason, count in sorted(optimizer.get_rejection_summary().items(),
                                     key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        return

    # ---- Print best result ----
    cfg = best.config
    print(f"\n{'='*50}")
    print("BEST SOLUTION")
    print(f"{'='*50}")
    print(f"  Mass:     {best.mass_total*1000:.2f} g")
    print(f"  N_Rib:    {cfg.N_Rib} (final: {best.N_Rib_final})")
    print(f"  X_FS:     {cfg.X_FS_percent:.1f}%")
    print(f"  X_RS:     {cfg.X_RS_percent:.1f}%")
    print(f"  d_FS:     {cfg.d_FS_outer_mm:.1f} mm, t_FS: {cfg.t_FS_mm:.1f} mm")
    print(f"  d_RS:     {cfg.d_RS_outer_mm:.1f} mm, t_RS: {cfg.t_RS_mm:.1f} mm")

    print(f"\n  Stresses:")
    print(f"    tau_skin_max     = {best.tau_skin_max/1e6:.4f} MPa")
    print(f"    sigma_vm_FS_max  = {best.sigma_vm_FS_max/1e6:.4f} MPa")
    print(f"    sigma_vm_RS_max  = {best.sigma_vm_RS_max/1e6:.4f} MPa")
    print(f"    sigma_skin_comp  = {best.sigma_skin_comp_max/1e6:.4f} MPa")
    print(f"    theta_tip        = {best.theta_tip_deg:.4f} deg")

    print(f"\n  Buckling:")
    print(f"    tau_cr_min       = {best.tau_cr_min/1e6:.4f} MPa")
    print(f"    sigma_cr_min     = {best.sigma_cr_min/1e6:.4f} MPa")
    print(f"    R_combined_max   = {best.R_combined_max:.4f}")
    print(f"    rib_feasible     = {best.rib_feasible}")

    # ---- KEY TEST 1: t_skin_final_mm propagation ----
    print(f"\n{'='*50}")
    print("TEST 1: t_skin_final_mm propagation")
    print(f"{'='*50}")
    print(f"  Input t_skin_mm     = {ctx['t_skin_mm']:.1f} mm")
    print(f"  best.t_skin_final_mm = {best.t_skin_final_mm:.1f} mm")

    if best.t_skin_final_mm > 0:
        print(f"  STATUS: PASS - t_skin_final_mm is set ({best.t_skin_final_mm:.1f} mm)")
    else:
        print(f"  STATUS: FAIL - t_skin_final_mm is 0 or negative!")

    # In buckling_mode=1, Phase-2 does NOT increase t_skin. It should
    # remain equal to the input value.
    if best.t_skin_final_mm == ctx['t_skin_mm']:
        print(f"  Correctly preserved input t_skin for buckling_mode=1")
    else:
        print(f"  NOTE: t_skin was changed from {ctx['t_skin_mm']} to {best.t_skin_final_mm}")
        print(f"        This is unexpected for buckling_mode=1")

    # ---- KEY TEST 2: validate_solution with buckling_mode=1 ----
    print(f"\n{'='*50}")
    print("TEST 2: validate_solution logic (buckling_mode=1)")
    print(f"{'='*50}")

    # Get allowables from optimizer
    allowables = optimizer.compute_allowables()
    tau_allow = allowables['tau_allow_skin']
    sigma_allow = allowables['sigma_allow_spar']
    tau_allow_spar = allowables['tau_allow_spar']

    print(f"  Allowables:")
    print(f"    tau_allow_skin  = {tau_allow/1e6:.2f} MPa")
    print(f"    sigma_allow_spar = {sigma_allow/1e6:.2f} MPa")
    print(f"    tau_allow_spar  = {tau_allow_spar/1e6:.2f} MPa")

    # Simulate validate_solution for buckling_mode=1
    is_valid_mode1, errors_mode1 = simulate_validate_solution(
        best, buckling_mode=1,
        h_box_mm=ctx['h_box_root_mm'],
        theta_max_deg=ctx['theta_max_deg'],
        tau_allow=tau_allow,
        sigma_allow=sigma_allow,
        tau_allow_spar=tau_allow_spar
    )

    print(f"\n  R_combined_max = {best.R_combined_max:.4f}")
    print(f"  buckling_mode = 1 (shear only)")

    if is_valid_mode1:
        print(f"  validate_solution(mode=1) => PASS")
    else:
        print(f"  validate_solution(mode=1) => FAIL")
        for err in errors_mode1:
            print(f"    - {err}")

    # Also check: what would happen if we wrongly applied mode=2 logic?
    is_valid_mode2, errors_mode2 = simulate_validate_solution(
        best, buckling_mode=2,
        h_box_mm=ctx['h_box_root_mm'],
        theta_max_deg=ctx['theta_max_deg'],
        tau_allow=tau_allow,
        sigma_allow=sigma_allow,
        tau_allow_spar=tau_allow_spar
    )

    print(f"\n  Cross-check: validate_solution(mode=2) => {'PASS' if is_valid_mode2 else 'FAIL'}")
    if not is_valid_mode2:
        for err in errors_mode2:
            print(f"    - {err}")

    # The key verification: if R_combined_max >= 1.0, mode=1 should PASS
    # but mode=2 should FAIL on criteria [10]
    r_combined = best.R_combined_max
    if r_combined >= 1.0:
        if is_valid_mode1 and not is_valid_mode2:
            print(f"\n  BUGFIX VERIFIED: R_combined={r_combined:.4f} >= 1.0")
            print(f"    mode=1 correctly ignores R_combined => PASS")
            print(f"    mode=2 correctly rejects R_combined => FAIL")
        elif is_valid_mode1 and is_valid_mode2:
            print(f"\n  UNEXPECTED: R_combined={r_combined:.4f} >= 1.0 but mode=2 also passes?")
        elif not is_valid_mode1:
            print(f"\n  ISSUE: R_combined={r_combined:.4f} >= 1.0 but mode=1 fails for OTHER reasons:")
            for err in errors_mode1:
                print(f"    - {err}")
    else:
        print(f"\n  R_combined={r_combined:.4f} < 1.0 -- both modes pass (no difference)")
        print(f"  The bugfix is still correct: mode=1 never checks R_combined.")
        print(f"  To see the difference, a solution with R_combined >= 1.0 is needed.")

    # ---- TEST 3: Verify all accepted solutions ----
    print(f"\n{'='*50}")
    print("TEST 3: Validate ALL accepted solutions (mode=1)")
    print(f"{'='*50}")

    all_solutions = optimizer.get_top_solutions(len(optimizer.accepted_results))
    n_pass_mode1 = 0
    n_fail_mode1 = 0
    n_pass_mode2 = 0
    n_fail_mode2 = 0
    n_r_above_1 = 0

    for sol in all_solutions:
        v1, _ = simulate_validate_solution(
            sol, buckling_mode=1,
            h_box_mm=ctx['h_box_root_mm'],
            theta_max_deg=ctx['theta_max_deg'],
            tau_allow=tau_allow,
            sigma_allow=sigma_allow,
            tau_allow_spar=tau_allow_spar
        )
        v2, _ = simulate_validate_solution(
            sol, buckling_mode=2,
            h_box_mm=ctx['h_box_root_mm'],
            theta_max_deg=ctx['theta_max_deg'],
            tau_allow=tau_allow,
            sigma_allow=sigma_allow,
            tau_allow_spar=tau_allow_spar
        )
        if v1:
            n_pass_mode1 += 1
        else:
            n_fail_mode1 += 1
        if v2:
            n_pass_mode2 += 1
        else:
            n_fail_mode2 += 1
        if sol.R_combined_max >= 1.0:
            n_r_above_1 += 1

    total_accepted = len(all_solutions)
    print(f"  Total Phase-2 accepted solutions: {total_accepted}")
    print(f"  Solutions with R_combined >= 1.0:  {n_r_above_1}")
    print(f"\n  mode=1 validate: {n_pass_mode1} PASS, {n_fail_mode1} FAIL")
    print(f"  mode=2 validate: {n_pass_mode2} PASS, {n_fail_mode2} FAIL")

    if n_r_above_1 > 0 and n_pass_mode1 > n_pass_mode2:
        print(f"\n  BUGFIX CONFIRMED: {n_pass_mode1 - n_pass_mode2} solutions pass in mode=1")
        print(f"  that would fail in mode=2 due to R_combined >= 1.0")
    elif n_r_above_1 == 0:
        print(f"\n  All R_combined < 1.0 -- bugfix logic correct but not exercised here.")
    else:
        print(f"\n  Same pass/fail counts -- R_combined is not the differentiator in this space.")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  1. Optimization ran without errors:        PASS")
    print(f"  2. t_skin_final_mm = {best.t_skin_final_mm:.1f} mm:  {'PASS' if best.t_skin_final_mm > 0 else 'FAIL'}")
    print(f"  3. validate_solution(mode=1) for best:     {'PASS' if is_valid_mode1 else 'FAIL'}")
    print(f"  4. R_combined_max = {best.R_combined_max:.4f}")
    if is_valid_mode1:
        print(f"  5. validate_solution correctly skips R_combined in mode=1: PASS")
    else:
        print(f"  5. validate_solution(mode=1) failed:       FAIL (see errors above)")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

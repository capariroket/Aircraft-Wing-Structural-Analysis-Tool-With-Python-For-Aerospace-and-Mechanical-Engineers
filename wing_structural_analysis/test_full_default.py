#!/usr/bin/env python3
"""
test_full_default.py - Full Default Parameters Test Script

Tests the optimization pipeline with default parameters from main.py.
Two test cases:
  1. buckling_mode=1 (shear only), load_dist='uniform'
  2. buckling_mode=2 (shear+compression), load_dist='uniform'

Material IDs: spar=3 (Carbon Fiber UD), skin=5 (PLA), rib=5 (PLA)
"""

import sys
import os
import time
import traceback
import signal
import numpy as np

# Add project root to path
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
# TIMEOUT HANDLER
# =============================================================================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out!")

# Only set alarm on Unix
if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, timeout_handler)


# =============================================================================
# DEFAULT PARAMETERS (from main.py)
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
    'rib_profile': 1,  # Rectangular
    'material_spar_id': 3,  # CFRP_UD
    'material_rib_id': 5,   # PLA
    'material_skin_id': 5,  # PLA
}


def setup_common():
    """Setup common objects from default parameters. Returns dict of all objects."""

    # Unpack defaults
    C_m = DEFAULTS['C_m']
    AR = DEFAULTS['AR']
    lambda_taper = DEFAULTS['lambda_taper']
    V_c = DEFAULTS['V_c']
    n = DEFAULTS['n']
    S_ref = DEFAULTS['S_ref']
    W_0 = DEFAULTS['W_0']
    rho_air = DEFAULTS['rho_air']
    x_ac_percent = DEFAULTS['x_ac_percent']
    Lambda_ac_deg = DEFAULTS['Lambda_ac_deg']
    b = DEFAULTS['b']
    t_over_c = DEFAULTS['t_over_c']

    # Computed geometry
    C_r = 2 * S_ref / (b * (1 + lambda_taper))  # [m]
    C_r_mm = C_r * 1000
    c_MGC = (4 / 3) * np.sqrt(S_ref / AR) * (
        (1 + lambda_taper + lambda_taper**2) /
        (1 + 2 * lambda_taper + lambda_taper**2)
    )
    Y_bar_mm = (b / 6) * ((1 + 2 * lambda_taper) / (1 + lambda_taper)) * 1000
    N_Rib = int(np.ceil(1 + np.sqrt(AR * S_ref) / c_MGC))
    L_span = b / 2
    c_tip = lambda_taper * C_r
    h_box_root = t_over_c * C_r
    h_box_root_mm = t_over_c * C_r_mm

    print(f"\n{'='*60}")
    print("COMPUTED GEOMETRY")
    print(f"{'='*60}")
    print(f"  b = {b} m")
    print(f"  L_span = {L_span:.4f} m = {L_span*1000:.2f} mm")
    print(f"  C_r = {C_r_mm:.2f} mm ({C_r:.4f} m)")
    print(f"  c_tip = {c_tip*1000:.2f} mm")
    print(f"  c_MGC = {c_MGC*1000:.2f} mm ({c_MGC:.4f} m)")
    print(f"  Y_bar = {Y_bar_mm:.2f} mm")
    print(f"  h_box_root = {h_box_root*1000:.2f} mm")
    print(f"  N_Rib = {N_Rib}")

    # Materials
    db = MaterialDatabase()
    mat_keys = list(db.materials.keys())
    mat_spar_key = mat_keys[DEFAULTS['material_spar_id'] - 1]
    mat_rib_key = mat_keys[DEFAULTS['material_rib_id'] - 1]
    mat_skin_key = mat_keys[DEFAULTS['material_skin_id'] - 1]
    materials = MaterialSelection.from_database(db, mat_spar_key, mat_skin_key, mat_rib_key)

    print(f"\n  Materials:")
    print(f"    Spar: {materials.spar.name} (E={materials.spar.E/1e9:.1f} GPa, sigma_u={materials.spar.sigma_u/1e6:.0f} MPa)")
    print(f"    Skin: {materials.skin.name} (E={materials.skin.E/1e9:.1f} GPa, sigma_u={materials.skin.sigma_u/1e6:.0f} MPa)")
    print(f"    Rib:  {materials.rib.name} (E={materials.rib.E/1e9:.1f} GPa)")

    # Objects
    planform = PlanformParams.from_input(
        b=b, AR=AR, taper_ratio=lambda_taper, t_c=t_over_c,
        S_ref=S_ref, C_r_mm=C_r_mm, c_MGC=c_MGC, Y_bar_mm=Y_bar_mm
    )

    flight = FlightCondition(
        W0=W_0, n=n, V_c=V_c, rho=rho_air,
        C_m=C_m, S_ref=S_ref, c_MGC=c_MGC
    )

    aero_center = AeroCenter(x_ac_percent=x_ac_percent, Lambda_ac_deg=Lambda_ac_deg)

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


def build_design_space(N_Rib):
    """Build design space with default ranges."""
    return DesignSpace(
        N_Rib=DesignRange(min_val=N_Rib, max_val=N_Rib, step=1),
        t_rib_mm=DesignRange(1.0, 5.0, 1.0),
        X_FS_percent=DesignRange(15.0, 30.0, 5.0),
        X_RS_percent=DesignRange(40.0, 65.0, 5.0),
        d_FS_outer_mm=DesignRange(16.0, 30.0, 2.0),
        t_FS_mm=DesignRange(0.5, 3.0, 0.5),
        d_RS_outer_mm=DesignRange(10.0, 25.0, 5.0),
        t_RS_mm=DesignRange(0.5, 3.0, 0.5),
    )


def print_top_solutions(optimizer, n=20):
    """Print top N solutions from optimizer."""
    top = optimizer.get_top_solutions(n)
    if not top:
        print("  No solutions found!")
        return

    print(f"\n  Top-{min(n, len(top))} Solutions (by mass):")
    print(f"  {'#':>3} {'Mass[g]':>8} {'N_Rib':>5} {'N_fin':>5} "
          f"{'X_FS':>5} {'X_RS':>5} "
          f"{'d_FS':>5} {'t_FS':>4} "
          f"{'d_RS':>5} {'t_RS':>4} "
          f"{'R_max':>6} {'t_sk':>5} {'Feasible':>8}")
    print(f"  {'-'*90}")

    for i, sol in enumerate(top, 1):
        cfg = sol.config
        print(f"  {i:3d} {sol.mass_total*1000:8.2f} "
              f"{cfg.N_Rib:5d} {sol.N_Rib_final:5d} "
              f"{cfg.X_FS_percent:5.1f} {cfg.X_RS_percent:5.1f} "
              f"{cfg.d_FS_outer_mm:5.1f} {cfg.t_FS_mm:4.1f} "
              f"{cfg.d_RS_outer_mm:5.1f} {cfg.t_RS_mm:4.1f} "
              f"{sol.R_combined_max:6.3f} "
              f"{sol.t_skin_final_mm:5.1f} "
              f"{'YES' if sol.rib_feasible else 'NO':>8}")


def print_best_detail(best):
    """Print detailed info about the best solution."""
    if best is None:
        print("  No solution found!")
        return

    cfg = best.config
    print(f"\n  {'='*50}")
    print(f"  BEST SOLUTION DETAILS")
    print(f"  {'='*50}")
    print(f"  Configuration:")
    print(f"    N_Rib = {cfg.N_Rib} (final: {best.N_Rib_final})")
    print(f"    t_rib = {cfg.t_rib_mm:.1f} mm")
    print(f"    X_FS = {cfg.X_FS_percent:.1f}%, X_RS = {cfg.X_RS_percent:.1f}%")
    print(f"    d_FS = {cfg.d_FS_outer_mm:.1f} mm, t_FS = {cfg.t_FS_mm:.1f} mm")
    print(f"    d_RS = {cfg.d_RS_outer_mm:.1f} mm, t_RS = {cfg.t_RS_mm:.1f} mm")

    print(f"\n  Mass Breakdown:")
    print(f"    Skin:  {best.mass_skin*1000:.2f} g")
    print(f"    FS:    {best.mass_FS*1000:.2f} g")
    print(f"    RS:    {best.mass_RS*1000:.2f} g")
    print(f"    Ribs:  {best.mass_ribs*1000:.2f} g")
    print(f"    Total: {best.mass_total*1000:.2f} g")

    print(f"\n  Stresses:")
    print(f"    tau_skin_max     = {best.tau_skin_max/1e6:.4f} MPa")
    print(f"    sigma_vm_FS_max  = {best.sigma_vm_FS_max/1e6:.4f} MPa")
    print(f"    sigma_vm_RS_max  = {best.sigma_vm_RS_max/1e6:.4f} MPa")
    print(f"    sigma_skin_comp  = {best.sigma_skin_comp_max/1e6:.4f} MPa")
    print(f"    theta_tip        = {best.theta_tip_deg:.4f} deg")

    print(f"\n  Box Beam:")
    print(f"    I_box_root     = {best.I_box_root*1e12:.2f} mm^4")
    print(f"    I_FS           = {best.I_FS*1e12:.2f} mm^4")
    print(f"    I_RS           = {best.I_RS*1e12:.2f} mm^4")
    print(f"    Skin fraction  = {best.skin_fraction*100:.1f}%")

    print(f"\n  Buckling:")
    print(f"    tau_cr_min     = {best.tau_cr_min/1e6:.4f} MPa")
    print(f"    sigma_cr_min   = {best.sigma_cr_min/1e6:.4f} MPa")
    print(f"    R_combined_max = {best.R_combined_max:.4f}")
    print(f"    rib_feasible   = {best.rib_feasible}")
    print(f"    t_skin_final   = {best.t_skin_final_mm:.1f} mm")

    print(f"\n  Load Sharing:")
    print(f"    eta_FS = {best.eta_FS:.4f}")
    print(f"    eta_RS = {best.eta_RS:.4f}")

    print(f"\n  Areas:")
    print(f"    A_Act_FS = {best.A_Act_FS*1e6:.4f} mm^2")
    print(f"    A_Cri_FS = {best.A_Cri_FS*1e6:.4f} mm^2")
    print(f"    A_Act_RS = {best.A_Act_RS*1e6:.4f} mm^2")
    print(f"    A_Cri_RS = {best.A_Cri_RS*1e6:.4f} mm^2")
    print(f"    FS margin = {(best.A_Act_FS/best.A_Cri_FS - 1)*100:.1f}%" if best.A_Cri_FS > 0 else "    FS A_Cri = 0")
    print(f"    RS margin = {(best.A_Act_RS/best.A_Cri_RS - 1)*100:.1f}%" if best.A_Cri_RS > 0 else "    RS A_Cri = 0")


def print_rejection_summary(optimizer):
    """Print rejection breakdown."""
    summary = optimizer.get_rejection_summary()
    if not summary:
        print("  No rejections!")
        return

    total = sum(summary.values())
    print(f"\n  Rejection Summary ({total} total):")
    for reason, count in sorted(summary.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {pct:5.1f}% | {count:6d} | {reason}")


# =============================================================================
# TEST 1: buckling_mode=1 (shear only), uniform load
# =============================================================================
def test_shear_only():
    """Test 1: Shear-only buckling mode with uniform load distribution."""
    print("\n")
    print("=" * 70)
    print("TEST 1: SHEAR ONLY BUCKLING (mode=1), UNIFORM LOAD")
    print("=" * 70)

    ctx = setup_common()
    ds = build_design_space(ctx['N_Rib'])

    print(f"\n  Design Space:")
    print(f"    N_Rib: {ctx['N_Rib']} (fixed)")
    print(f"    t_rib: 1.0 to 5.0 step 1.0 ({ds.t_rib_mm.n_values} values)")
    print(f"    X_FS:  15 to 30 step 5 ({ds.X_FS_percent.n_values} values)")
    print(f"    X_RS:  40 to 65 step 5 ({ds.X_RS_percent.n_values} values)")
    print(f"    d_FS:  16 to 30 step 2 ({ds.d_FS_outer_mm.n_values} values)")
    print(f"    t_FS:  0.5 to 3.0 step 0.5 ({ds.t_FS_mm.n_values} values)")
    print(f"    d_RS:  10 to 25 step 5 ({ds.d_RS_outer_mm.n_values} values)")
    print(f"    t_RS:  0.5 to 3.0 step 0.5 ({ds.t_RS_mm.n_values} values)")
    print(f"    Total combinations: {ds.total_combinations}")

    # Set timeout
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(300)

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
            rib_parabolic=False,  # Rectangular profile
            load_dist='uniform',
            pitch_dist='uniform',
            n_workers=1,
            s_min_mm=ctx['s_min_mm'],
            buckling_mode=1,
        )

        elapsed = time.time() - t0

        print(f"\n  Optimization completed in {elapsed:.1f}s")
        print(f"\n  Phase-1 Results:")
        print(f"    Total evaluated: {len(optimizer.results)}")
        print(f"    Valid geometry:  {sum(1 for r in optimizer.results if r.valid)}")
        print(f"    Phase-1 accepted: {len([r for r in optimizer.results if r.accepted])}")

        # Note: after Phase-2, accepted_results is overwritten with Phase-2 feasible
        print(f"    Phase-2 accepted (final): {len(optimizer.accepted_results)}")

        print_rejection_summary(optimizer)
        print_top_solutions(optimizer, n=20)
        print_best_detail(best)

        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        return best, optimizer

    except TimeoutError:
        print(f"\n  TIMEOUT after 300 seconds!")
        return None, None
    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        return None, None


# =============================================================================
# TEST 2: buckling_mode=2 (shear+compression), uniform load
# =============================================================================
def test_shear_compression():
    """Test 2: Shear+compression buckling mode with uniform load distribution."""
    print("\n")
    print("=" * 70)
    print("TEST 2: SHEAR + COMPRESSION BUCKLING (mode=2), UNIFORM LOAD")
    print("=" * 70)

    ctx = setup_common()
    ds = build_design_space(ctx['N_Rib'])

    print(f"\n  Design Space: {ds.total_combinations} combinations")
    print(f"  Buckling mode 2 enables t_skin increase in Phase-2")

    # Set timeout
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(300)

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
            buckling_mode=2,
        )

        elapsed = time.time() - t0

        print(f"\n  Optimization completed in {elapsed:.1f}s")
        print(f"\n  Phase-1 Results:")
        print(f"    Total evaluated: {len(optimizer.results)}")
        print(f"    Valid geometry:  {sum(1 for r in optimizer.results if r.valid)}")

        # Count Phase-1 accepted (before Phase-2 overwrites)
        # We can infer from total - rejected
        total_rejected = sum(optimizer.get_rejection_summary().values())
        total_valid = sum(1 for r in optimizer.results if r.valid)
        phase1_accepted_approx = len(optimizer.results) - total_rejected

        print(f"    Phase-2 accepted (final): {len(optimizer.accepted_results)}")

        # Count UNRESOLVABLE
        unresolvable_count = optimizer.get_rejection_summary().get('UNRESOLVABLE_BUCKLING', 0)
        print(f"    UNRESOLVABLE_BUCKLING: {unresolvable_count}")

        print_rejection_summary(optimizer)

        # Check t_skin increases in Phase-2
        if optimizer.accepted_results:
            t_skin_values = set()
            for sol in optimizer.accepted_results:
                t_skin_values.add(sol.t_skin_final_mm)
            print(f"\n  t_skin values in accepted solutions: {sorted(t_skin_values)}")
            if len(t_skin_values) > 1 or (len(t_skin_values) == 1 and list(t_skin_values)[0] > ctx['t_skin_mm']):
                print(f"  Phase-2 t_skin increase: YES (original: {ctx['t_skin_mm']} mm)")
            else:
                print(f"  Phase-2 t_skin increase: NO (all at {ctx['t_skin_mm']} mm)")

        print_top_solutions(optimizer, n=20)
        print_best_detail(best)

        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        return best, optimizer

    except TimeoutError:
        print(f"\n  TIMEOUT after 300 seconds!")
        return None, None
    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        return None, None


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("WING STRUCTURAL ANALYSIS - FULL DEFAULT PARAMETER TEST")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Pre-flight: verify imports and materials
    print("\nPre-flight checks...")
    db = MaterialDatabase()
    mat_keys = list(db.materials.keys())
    print(f"  Materials available ({len(mat_keys)}): {mat_keys}")
    print(f"  Spar (ID=3): {mat_keys[2]} = {db.get_material(mat_keys[2]).name}")
    print(f"  Skin (ID=5): {mat_keys[4]} = {db.get_material(mat_keys[4]).name}")
    print(f"  Rib  (ID=5): {mat_keys[4]} = {db.get_material(mat_keys[4]).name}")

    total_start = time.time()

    # Test 1
    best1, opt1 = test_shear_only()

    # Test 2
    best2, opt2 = test_shear_compression()

    # Summary
    total_elapsed = time.time() - total_start
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total elapsed time: {total_elapsed:.1f}s")

    if best1:
        print(f"\n  Test 1 (Shear Only):")
        print(f"    Best mass: {best1.mass_total*1000:.2f} g")
        print(f"    N_Rib_final: {best1.N_Rib_final}")
        print(f"    R_combined: {best1.R_combined_max:.4f}")
        print(f"    t_skin_final: {best1.t_skin_final_mm:.1f} mm")
    else:
        print(f"\n  Test 1 (Shear Only): FAILED or NO SOLUTION")

    if best2:
        print(f"\n  Test 2 (Shear+Compression):")
        print(f"    Best mass: {best2.mass_total*1000:.2f} g")
        print(f"    N_Rib_final: {best2.N_Rib_final}")
        print(f"    R_combined: {best2.R_combined_max:.4f}")
        print(f"    t_skin_final: {best2.t_skin_final_mm:.1f} mm")
    else:
        print(f"\n  Test 2 (Shear+Compression): FAILED or NO SOLUTION")

    if opt2:
        unresolvable = opt2.get_rejection_summary().get('UNRESOLVABLE_BUCKLING', 0)
        print(f"    UNRESOLVABLE_BUCKLING count: {unresolvable}")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")

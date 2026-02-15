"""
test_pipeline.py - Phase-1 + Phase-2 Pipeline Smoke Test

Small design space (3-4 combinations) with buckling_mode=1 (shear only).
Validates that the GridSearchOptimizer runs end-to-end without errors.
"""

import sys
import numpy as np

# --- Setup paths ---
sys.path.insert(0, '/Users/apple/Desktop/dosyalar/wing_structural_analysis')

from materials import MaterialDatabase, MaterialSelection
from geometry import PlanformParams
from loads import FlightCondition, AeroCenter
from optimization import (DesignRange, DesignSpace, OptimizationConfig,
                           GridSearchOptimizer, run_optimization)


def make_defaults():
    """Create default planform, flight, aero center, materials."""
    planform = PlanformParams.from_input(
        b=2.0, AR=8.0, taper_ratio=0.5, t_c=0.12,
        S_ref=0.5, C_r_mm=300, c_MGC=0.25, Y_bar_mm=400
    )

    flight = FlightCondition(
        W0=50, n=2.0, V_c=25, rho=1.225,
        C_m=-0.05, S_ref=0.5, c_MGC=0.25
    )

    ac = AeroCenter(x_ac_percent=25, Lambda_ac_deg=0)

    db = MaterialDatabase()
    materials = MaterialSelection.from_database(db, 'AL7075-T6', 'AL7075-T6', 'PLA')

    return planform, flight, ac, materials


def make_small_design_space():
    """Create a tiny design space with ~3-4 effective combinations.

    Most parameters are fixed (single value), only d_FS and d_RS vary.
    """
    return DesignSpace(
        N_Rib=DesignRange(5, 5, 1),            # fixed: 5
        t_rib_mm=DesignRange(2.0, 2.0, 1.0),   # fixed: 2.0
        X_FS_percent=DesignRange(25, 25, 1),    # fixed: 25%
        X_RS_percent=DesignRange(65, 65, 1),    # fixed: 65%
        d_FS_outer_mm=DesignRange(18, 20, 2),   # 18, 20 -> 2 values
        t_FS_mm=DesignRange(2.0, 2.0, 1.0),    # fixed: 2.0
        d_RS_outer_mm=DesignRange(14, 16, 2),   # 14, 16 -> 2 values
        t_RS_mm=DesignRange(1.5, 1.5, 1.0),    # fixed: 1.5
    )


def test_phase1_phase2():
    """Run Phase-1 + Phase-2 with a small design space."""
    print("=" * 60)
    print("TEST: Phase-1 + Phase-2 Pipeline (buckling_mode=1)")
    print("=" * 60)

    planform, flight, ac, materials = make_defaults()
    design_space = make_small_design_space()

    n_combos = design_space.total_combinations
    print(f"\nDesign space: {n_combos} combinations")
    assert n_combos <= 10, f"Design space too large: {n_combos}"

    # Run the full two-phase optimization
    # Note: top_n is a parameter of optimizer.run(), not run_optimization().
    # run_optimization uses the default top_n=20 internally.
    best, optimizer = run_optimization(
        planform, flight, ac, materials, design_space,
        t_skin_mm=1.0,
        SF=1.5,
        buckling_mode=1,   # shear only
        s_min_mm=20.0,
        n_workers=1,
    )

    # --- Verify Phase-1 ---
    print(f"\n--- Phase-1 Results ---")
    print(f"  Total evaluated: {len(optimizer.results)}")
    print(f"  Phase-1 accepted: {len(optimizer.accepted_results)}")
    print(f"  Rejections: {optimizer.get_rejection_summary()}")

    assert len(optimizer.results) == n_combos, (
        f"Expected {n_combos} evaluations, got {len(optimizer.results)}")

    # --- Verify Phase-2 ---
    if best is None:
        print("\n  No feasible solution found (expected for some material combos).")
        print("  Checking that pipeline did not crash: PASS")
    else:
        print(f"\n--- Best Solution ---")
        cfg = best.config
        print(f"  d_FS = {cfg.d_FS_outer_mm} mm, t_FS = {cfg.t_FS_mm} mm")
        print(f"  d_RS = {cfg.d_RS_outer_mm} mm, t_RS = {cfg.t_RS_mm} mm")
        print(f"  N_Rib_initial = {cfg.N_Rib}, N_Rib_final = {best.N_Rib_final}")
        print(f"  t_skin_final = {best.t_skin_final_mm:.1f} mm")
        print(f"  mass_total = {best.mass_total * 1000:.2f} g")
        print(f"  tau_skin_max = {best.tau_skin_max / 1e6:.4f} MPa")
        print(f"  sigma_vm_FS = {best.sigma_vm_FS_max / 1e6:.2f} MPa")
        print(f"  sigma_vm_RS = {best.sigma_vm_RS_max / 1e6:.2f} MPa")
        print(f"  R_combined_max = {best.R_combined_max:.4f}")
        print(f"  rib_feasible = {best.rib_feasible}")

        # Sanity checks on best solution
        assert best.accepted, "Best solution should be accepted"
        assert best.mass_total > 0, "Mass should be positive"
        assert best.rib_feasible, "Best should be rib-feasible"
        assert best.R_combined_max < 1.0 or best.R_combined_max == 0.0, \
            f"R_combined should be < 1.0, got {best.R_combined_max}"

        if best.final_y_ribs is not None:
            print(f"  final_y_ribs: {len(best.final_y_ribs)} stations")
            print(f"    range: [{best.final_y_ribs[0]*1000:.1f}, "
                  f"{best.final_y_ribs[-1]*1000:.1f}] mm")

    print("\nPipeline test: PASS")
    return best, optimizer


def test_evaluation_result_dataclass():
    """Test that EvaluationResult with numpy array default works."""
    from optimization import EvaluationResult, ConfigCandidate

    print("\n" + "=" * 60)
    print("TEST: EvaluationResult dataclass (numpy default)")
    print("=" * 60)

    cfg = ConfigCandidate(
        N_Rib=5, t_rib_mm=2.0,
        X_FS_percent=25, X_RS_percent=65,
        d_FS_outer_mm=18, t_FS_mm=2.0,
        d_RS_outer_mm=14, t_RS_mm=1.5
    )

    # Create two instances, verify no shared mutable state
    r1 = EvaluationResult(config=cfg, valid=True, accepted=True)
    r2 = EvaluationResult(config=cfg, valid=True, accepted=True)

    # Assign numpy array to one
    r1.final_y_ribs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Other should still be None
    assert r2.final_y_ribs is None, "Mutable default leak detected!"
    print("  No mutable default leak: PASS")

    # Verify default values
    assert r1.mass_total == 0.0
    assert r1.N_Rib_final == 0
    assert r1.rib_feasible is True
    print("  Default values correct: PASS")

    print("Dataclass test: PASS")


if __name__ == "__main__":
    # Test 1: Dataclass correctness
    test_evaluation_result_dataclass()

    # Test 2: Full pipeline
    best, optimizer = test_phase1_phase2()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

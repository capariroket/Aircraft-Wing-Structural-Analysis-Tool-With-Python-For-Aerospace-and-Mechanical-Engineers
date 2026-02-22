"""
ga_optimization.py - Genetic Algorithm Optimization Module

Uses pymoo's Mixed-Variable GA for single-objective (min mass)
wing structural optimization with 10 inequality constraints.

Constraint handling: Deb's feasibility rules (built into pymoo)
- Feasible always beats infeasible
- Among infeasible: less total constraint violation wins
- Among feasible: better objective wins

Design variables (8):
  1. N_Rib       [Integer]  - Number of ribs
  2. t_rib_mm    [Real]     - Rib thickness [mm]
  3. X_FS_percent [Real]    - Front spar position [% chord]
  4. X_RS_percent [Real]    - Rear spar position [% chord]
  5. d_FS_outer_mm [Real]   - Front spar outer diameter [mm]
  6. t_FS_mm     [Real]     - Front spar wall thickness [mm]
  7. d_RS_outer_mm [Real]   - Rear spar outer diameter [mm]
  8. t_RS_mm     [Real]     - Rear spar wall thickness [mm]

Constraints (10, g <= 0 = feasible):
  g[0]: tau_skin - tau_allow           (skin shear stress)
  g[1]: FS_interaction - 1.0           (front spar interaction)
  g[2]: RS_interaction - 1.0           (rear spar interaction)
  g[3]: A_Cri_FS - A_Act_FS           (front spar area)
  g[4]: A_Cri_RS - A_Act_RS           (rear spar area)
  g[5]: d_FS + 10 - h_box             (FS assembly clearance)
  g[6]: d_RS + 10 - h_box             (RS assembly clearance)
  g[7]: theta_tip - theta_max         (tip twist)
  g[8]: sigma_comp - sigma_cr         (compression buckling)
  g[9]: R_combined - 1.0              (combined interaction)
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

from loads import LoadDistributionType, PitchMomentDistributionType
from optimization import (OptimizationConfig, DesignSpace, ConfigCandidate,
                           EvaluationResult, GridSearchOptimizer)


@dataclass
class GAConfig:
    """Configuration for GA optimization."""
    pop_size: int = 100
    n_gen: int = 150
    seed: int = 42


# =============================================================================
# PROGRESS CALLBACK
# =============================================================================

class GAProgressCallback(Callback):
    """Print progress during GA optimization."""

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop

        # Count feasible individuals
        CV = pop.get("CV")
        feasible = int(np.sum(CV <= 0)) if CV is not None else 0

        # Best feasible objective
        F = pop.get("F")
        best_f = float('inf')
        if CV is not None and F is not None:
            feas_mask = (CV.flatten() <= 0)
            if np.any(feas_mask):
                best_f = float(np.min(F[feas_mask]))

        elapsed = time.time() - self.start_time

        if gen % 10 == 0 or gen <= 5:
            if best_f < float('inf'):
                print(f"  Gen {gen:4d}: {feasible}/{len(pop)} feasible, "
                      f"best = {best_f*1000:.2f} g, elapsed = {elapsed:.1f}s")
            else:
                print(f"  Gen {gen:4d}: {feasible}/{len(pop)} feasible, "
                      f"no feasible yet, elapsed = {elapsed:.1f}s")


# =============================================================================
# PYMOO PROBLEM DEFINITION
# =============================================================================

class WingStructuralProblem(ElementwiseProblem):
    """pymoo problem for wing structural optimization.

    Single objective: minimize total mass
    10 inequality constraints with Deb's feasibility rules
    """

    def __init__(self, opt_config: OptimizationConfig, design_space: DesignSpace):
        self.opt_config = opt_config
        self.design_space = design_space

        # Reuse GridSearchOptimizer's evaluate_config (fail-fast pipeline)
        self.evaluator = GridSearchOptimizer(opt_config, design_space)

        # Tracking
        self.eval_count = 0
        self.feasible_count = 0
        self.best_mass = float('inf')
        self.accepted_results: List[EvaluationResult] = []

        # Pre-compute assembly height limits
        h_FS_mm = opt_config.h_FS_mm
        if h_FS_mm <= 0:
            h_FS_mm = opt_config.planform.t_c * opt_config.planform.C_r * 1000
        h_RS_mm = opt_config.h_RS_mm
        if h_RS_mm <= 0:
            h_RS_mm = opt_config.planform.t_c * opt_config.planform.C_r * 1000
        self._h_FS_mm = h_FS_mm
        self._h_RS_mm = h_RS_mm

        # Define mixed variables with bounds from design space
        vars_dict = {
            "N_Rib": Integer(bounds=(int(design_space.N_Rib.min_val),
                                     int(design_space.N_Rib.max_val))),
            "t_rib_mm": Real(bounds=(design_space.t_rib_mm.min_val,
                                     design_space.t_rib_mm.max_val)),
            "X_FS_percent": Real(bounds=(design_space.X_FS_percent.min_val,
                                         design_space.X_FS_percent.max_val)),
            "X_RS_percent": Real(bounds=(design_space.X_RS_percent.min_val,
                                         design_space.X_RS_percent.max_val)),
            "d_FS_outer_mm": Real(bounds=(design_space.d_FS_outer_mm.min_val,
                                          design_space.d_FS_outer_mm.max_val)),
            "t_FS_mm": Real(bounds=(design_space.t_FS_mm.min_val,
                                    design_space.t_FS_mm.max_val)),
            "d_RS_outer_mm": Real(bounds=(design_space.d_RS_outer_mm.min_val,
                                          design_space.d_RS_outer_mm.max_val)),
            "t_RS_mm": Real(bounds=(design_space.t_RS_mm.min_val,
                                    design_space.t_RS_mm.max_val)),
        }

        super().__init__(vars=vars_dict, n_obj=1, n_ieq_constr=10)

    def _evaluate(self, X, out, *args, **kwargs):
        self.eval_count += 1

        config = ConfigCandidate(
            N_Rib=int(X["N_Rib"]),
            t_rib_mm=float(X["t_rib_mm"]),
            X_FS_percent=float(X["X_FS_percent"]),
            X_RS_percent=float(X["X_RS_percent"]),
            d_FS_outer_mm=float(X["d_FS_outer_mm"]),
            t_FS_mm=float(X["t_FS_mm"]),
            d_RS_outer_mm=float(X["d_RS_outer_mm"]),
            t_RS_mm=float(X["t_RS_mm"]),
        )

        result = self.evaluator.evaluate_config(config)

        # Track accepted solutions
        if result.accepted:
            self.feasible_count += 1
            self.accepted_results.append(result)
            if result.mass_total < self.best_mass:
                self.best_mass = result.mass_total

        # Objective: mass (large penalty if not computed)
        mass = result.mass_total if result.mass_total > 0 else 100.0
        out["F"] = [mass]

        # 10 inequality constraints (g <= 0 = feasible)
        out["G"] = self._compute_constraints(config, result)

    def _compute_constraints(self, config: ConfigCandidate,
                             result: EvaluationResult) -> np.ndarray:
        """Compute constraint values from evaluation result.

        For constraints not computed (due to early rejection), use BIG penalty.
        This correctly guides Deb's rules: solutions passing more checks
        have lower total constraint violation.
        """
        g = np.zeros(10)
        BIG = 1e3

        allowables = self.evaluator._allowables

        # g[5], g[6]: Assembly - ALWAYS computable from config alone
        g[5] = config.d_FS_outer_mm + 10 - self._h_FS_mm
        g[6] = config.d_RS_outer_mm + 10 - self._h_RS_mm

        # Phase 1 results (tau_skin, twist)
        has_phase1 = (result.tau_skin_max > 0 or result.theta_tip_deg > 0)
        if has_phase1:
            g[0] = result.tau_skin_max - allowables['tau_allow_skin']
            g[7] = result.theta_tip_deg - self.opt_config.theta_max_deg
        else:
            g[0] = BIG
            g[7] = BIG

        # Phase 2 results (spar interaction, area)
        has_phase2 = (result.A_Act_FS > 0)
        if has_phase2:
            sigma_allow = allowables['sigma_allow_spar']

            # Von Mises criterion: sqrt(σ² + 3τ²) < σ_allow
            sigma_vm_FS = (result.sigma_b_FS_max**2 + 3*result.tau_FS_max**2)**0.5
            g[1] = sigma_vm_FS / sigma_allow - 1.0

            sigma_vm_RS = (result.sigma_b_RS_max**2 + 3*result.tau_RS_max**2)**0.5
            g[2] = sigma_vm_RS / sigma_allow - 1.0

            g[3] = result.A_Cri_FS - result.A_Act_FS
            g[4] = result.A_Cri_RS - result.A_Act_RS
        else:
            g[1] = BIG
            g[2] = BIG
            g[3] = BIG
            g[4] = BIG

        # Phase 3 results (buckling)
        has_phase3 = (result.sigma_cr_min > 0 or result.R_combined_max > 0)
        if has_phase3:
            g[8] = result.sigma_skin_comp_max - result.sigma_cr_min \
                if result.sigma_cr_min > 0 else BIG
            g[9] = result.R_combined_max - 1.0 \
                if result.R_combined_max > 0 else BIG
        else:
            g[8] = BIG
            g[9] = BIG

        return g


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_ga_optimization(planform, flight, aero_center, materials, design_space,
                        t_skin_mm=1.0, SF=1.5, h_FS_mm=0.0, h_RS_mm=0.0,
                        theta_max_deg=2.0, rib_parabolic=False,
                        load_dist="elliptic", pitch_dist="chord_weighted",
                        ga_config=None, s_min_mm=20.0, buckling_mode=1,
                        N_Rib_max_factor=2.0, t_skin_step_mm=0.3, Y_bar_m=None):
    """
    Run Genetic Algorithm optimization.

    Uses pymoo's MixedVariableGA with Deb's feasibility rules.
    Same interface as run_optimization() for easy swapping.

    Args:
        (same as run_optimization)
        ga_config: GAConfig with pop_size, n_gen, seed

    Returns:
        Tuple of (best_result, optimizer) - same interface as grid search
    """
    if ga_config is None:
        ga_config = GAConfig()

    load_dist_type = (LoadDistributionType.ELLIPTIC if load_dist == "elliptic"
                      else (LoadDistributionType.SIMPLIFIED if load_dist == "simplified"
                            else LoadDistributionType.UNIFORM))
    pitch_dist_type = (PitchMomentDistributionType.CHORD_WEIGHTED
                       if pitch_dist == "chord_weighted"
                       else PitchMomentDistributionType.UNIFORM)

    opt_config = OptimizationConfig(
        planform=planform,
        flight=flight,
        aero_center=aero_center,
        materials=materials,
        t_skin_mm=t_skin_mm,
        SF=SF,
        h_FS_mm=h_FS_mm,
        h_RS_mm=h_RS_mm,
        theta_max_deg=theta_max_deg,
        rib_parabolic=rib_parabolic,
        load_dist_type=load_dist_type,
        pitch_dist_type=pitch_dist_type,
        s_min_mm=s_min_mm,
        buckling_mode=buckling_mode,
        N_Rib_max_factor=N_Rib_max_factor,
        t_skin_step_mm=t_skin_step_mm,
        Y_bar_m=Y_bar_m,
    )

    problem = WingStructuralProblem(opt_config, design_space)
    algorithm = MixedVariableGA(pop_size=ga_config.pop_size)
    termination = get_termination("n_gen", ga_config.n_gen)
    callback = GAProgressCallback()

    print(f"\nStarting GA optimization...")
    print(f"  Population: {ga_config.pop_size}")
    print(f"  Generations: {ga_config.n_gen}")
    print(f"  Max evaluations: ~{ga_config.pop_size * ga_config.n_gen}")

    start_time = time.time()

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=ga_config.seed,
        verbose=False,
        callback=callback,
    )

    elapsed = time.time() - start_time

    print(f"\nGA complete in {elapsed:.1f}s")
    print(f"  Total evaluations: {problem.eval_count}")
    print(f"  Feasible found: {problem.feasible_count}")

    # Extract best solution
    best_result = None
    if res.X is not None:
        best_config = ConfigCandidate(
            N_Rib=int(res.X["N_Rib"]),
            t_rib_mm=float(res.X["t_rib_mm"]),
            X_FS_percent=float(res.X["X_FS_percent"]),
            X_RS_percent=float(res.X["X_RS_percent"]),
            d_FS_outer_mm=float(res.X["d_FS_outer_mm"]),
            t_FS_mm=float(res.X["t_FS_mm"]),
            d_RS_outer_mm=float(res.X["d_RS_outer_mm"]),
            t_RS_mm=float(res.X["t_RS_mm"]),
        )

        best_result = problem.evaluator.evaluate_config(best_config)

        if best_result.accepted:
            print(f"  Best mass: {best_result.mass_total*1000:.2f} g")
        else:
            print(f"  WARNING: Best GA solution infeasible ({best_result.rejection_reason})")
            # Fall back to best from accepted history
            if problem.accepted_results:
                best_result = min(problem.accepted_results,
                                  key=lambda r: r.mass_total)
                print(f"  Using best feasible from history: "
                      f"{best_result.mass_total*1000:.2f} g")
            else:
                best_result = None

    elif problem.accepted_results:
        best_result = min(problem.accepted_results, key=lambda r: r.mass_total)
        print(f"  Best from history: {best_result.mass_total*1000:.2f} g")

    if best_result is None:
        print("  WARNING: No feasible solution found!")

    # Return optimizer-compatible interface
    optimizer = problem.evaluator
    optimizer.accepted_results = problem.accepted_results
    optimizer.rejection_counts = {"GA mode": problem.eval_count - problem.feasible_count}

    return best_result, optimizer


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== GA Optimization Module Test ===\n")

    from geometry import PlanformParams
    from loads import FlightCondition, AeroCenter
    from materials import MaterialDatabase, MaterialSelection
    from optimization import DesignRange

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

    design_space = DesignSpace(
        N_Rib=DesignRange(4, 6, 1),
        t_rib_mm=DesignRange(1.5, 2.5, 0.5),
        X_FS_percent=DesignRange(20, 30, 5),
        X_RS_percent=DesignRange(60, 70, 5),
        d_FS_outer_mm=DesignRange(16, 22, 2),
        t_FS_mm=DesignRange(1.5, 2.5, 0.5),
        d_RS_outer_mm=DesignRange(14, 18, 2),
        t_RS_mm=DesignRange(1.0, 2.0, 0.5),
    )

    ga_config = GAConfig(pop_size=50, n_gen=50, seed=42)

    best, optimizer = run_ga_optimization(
        planform, flight, ac, materials, design_space,
        t_skin_mm=1.0, SF=1.5,
        ga_config=ga_config
    )

    if best:
        print(f"\n--- Best GA Configuration ---")
        print(f"  N_Rib = {best.config.N_Rib} (final: {best.N_Rib_final})")
        print(f"  X_FS = {best.config.X_FS_percent:.1f}%")
        print(f"  X_RS = {best.config.X_RS_percent:.1f}%")
        print(f"  d_FS = {best.config.d_FS_outer_mm:.1f} mm, t_FS = {best.config.t_FS_mm:.2f} mm")
        print(f"  d_RS = {best.config.d_RS_outer_mm:.1f} mm, t_RS = {best.config.t_RS_mm:.2f} mm")
        print(f"  Total mass: {best.mass_total*1000:.2f} g")
        print(f"  R_combined_max = {best.R_combined_max:.4f}")
        print(f"  Feasible: {best.accepted}")
    else:
        print("\nNo feasible solution found.")

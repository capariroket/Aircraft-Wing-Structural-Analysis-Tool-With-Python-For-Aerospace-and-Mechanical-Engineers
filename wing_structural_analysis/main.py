#!/usr/bin/env python3
"""
main.py - Half-Wing Structural Sizing Main Program

Interactive CLI for wing structural optimization.
Combines all modules and generates complete results.
"""

import numpy as np
from typing import Optional

from materials import MaterialDatabase, MaterialSelection
from geometry import (PlanformParams, SparPosition, compute_all_stations,
                      compute_spar_sweep_angle, compute_skin_arc_lengths_root,
                      compute_skin_area_half_wing, compute_rib_areas_root)
from loads import (FlightCondition, AeroCenter, analyze_loads,
                   LoadDistributionType, PitchMomentDistributionType,
                   compute_root_reactions)
from torsion import analyze_all_stations as analyze_torsion_stations, find_critical_station
from spars import (SparProperties, analyze_spars_all_stations, find_critical_spar_station,
                   spar_mass, LoadSharing, BoxBeamSection, compute_box_beam_section,
                   analyze_box_beam_stress)
from ribs import (RibProperties, generate_rib_geometries, compute_total_rib_mass,
                  format_rib_outputs, AdaptiveRibConfig, adaptive_rib_insertion,
                  compute_bay_results, format_bay_results_table, format_bay_results_csv,
                  compute_minimum_rib_count,
                  fix_rib_web_buckling, format_rib_web_buckling_table)
from scipy.interpolate import interp1d
from optimization import (DesignRange, DesignSpace, run_optimization,
                          GridSearchOptimizer, EvaluationResult)
from ga_optimization import run_ga_optimization, GAConfig
from reporting import (OptimalConfig, MassBreakdown, StressResults,
                       RootReactionsReport, OptimizationHistory,
                       generate_full_report, generate_mass_table,
                       generate_optimization_history_section,
                       check_for_warnings, export_to_json)
from plots import PlotConfig, generate_all_plots


# =============================================================================
# DEFAULT VALUES (from prompt specification)
# =============================================================================

DEFAULTS = {
    # Flight science
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

    # Geometry
    'b': 5.195,
    't_over_c': 0.17,
    # Note: C_r, c_MGC, Y_bar, h_box, N_Rib are now calculated from formulas, not user inputs

    # Material limits (legacy)
    'sigma_max_composite_MPa': 300.0,
    'tau_max_composite_MPa': 150.0,
    'sigma_max_PLA_MPa': 50.0,
    'tau_max_PLA_MPa': 25.0,
    'rho_Skin': 1200.0,
    'rho_Internal': 1200.0,

    # Structural
    't_skin_mm': 1.2,
    'SF': 1.0,
    'theta_max_deg': 2.0,  # Maximum allowable tip twist [degrees]
    's_min_mm': 20.0,      # Minimum practical rib spacing [mm]
    'N_Rib_max_factor': 2.0,  # N_Rib_max = factor * N_Rib_min
    't_skin_step_mm': 0.3,    # Phase-2 skin thickness increment [mm]
    'rib_profile': 1,  # 1=Rectangular, 2=Parabolic
    'load_dist_type': 1,   # 1=Uniform, 2=Elliptic
    'buckling_mode': 1,    # 1=Shear only, 2=Shear+Compression

    # Design ranges (default single values -> will be expanded to ranges)
    't_rib_mm': 3.0,
    'X_FS_percent': 22.5,
    'X_RS_percent': 52.5,
    'd_FS_outer_mm': 23.0,
    'd_RS_outer_mm': 17.5,
    't_FS_mm': 1.75,
    't_RS_mm': 1.75,

    # Material IDs
    'material_spar_id': 3,
    'material_rib_id': 5,
    'material_skin_id': 5,
}


# =============================================================================
# INPUT HELPERS
# =============================================================================

def get_input(prompt: str, default, value_type=float) -> any:
    """
    Get user input with default value.

    Args:
        prompt: Prompt to display
        default: Default value if user presses Enter
        value_type: Type to convert input to

    Returns:
        User input or default value
    """
    while True:
        try:
            user_input = input(f"{prompt} (default = {default}): ").strip()
            if user_input == "":
                return default
            return value_type(user_input)
        except ValueError:
            print(f"  Invalid input. Please enter a valid {value_type.__name__}.")


def get_range_input(name: str, unit: str, default_min: float, default_max: float, default_step: float) -> DesignRange:
    """
    Get min/max/step range from user.

    Args:
        name: Parameter name
        unit: Unit string
        default_min: Default minimum value
        default_max: Default maximum value
        default_step: Default step value

    Returns:
        DesignRange object
    """
    print(f"\n  {name} [{unit}]:")
    min_val = get_input(f"    min", default_min)
    max_val = get_input(f"    max", default_max)
    step = get_input(f"    step", default_step)

    if step <= 0:
        step = 1.0
    if max_val < min_val:
        max_val = min_val

    return DesignRange(min_val, max_val, step)


def print_material_table(db: MaterialDatabase):
    """Print available materials."""
    print("\nAvailable Materials:")
    print("-" * 50)
    materials = list(db.materials.items())
    for i, (key, mat) in enumerate(materials, 1):
        print(f"  {i}. {key}: {mat.name}")
        print(f"     E={mat.E/1e9:.1f} GPa, σ_u={mat.sigma_u/1e6:.0f} MPa, ρ={mat.density} kg/m³")
    print("-" * 50)
    return materials


def select_material(db: MaterialDatabase, prompt: str, default_id: int) -> str:
    """Select material by number."""
    materials = list(db.materials.keys())
    while True:
        idx = get_input(prompt, default_id, int)
        if 1 <= idx <= len(materials):
            return materials[idx - 1]
        print(f"  Please enter a number between 1 and {len(materials)}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    """Main program entry point."""
    print("=" * 70)
    print("HALF-WING RIB-SPAR-SKIN PRELIMINARY STRUCTURAL SIZING")
    print("with Torsion/Stress-Flow Analysis")
    print("=" * 70)

    # Initialize material database
    db = MaterialDatabase()

    # ==========================================================================
    # SECTION 1: FLIGHT SCIENCE / AERO PARAMETERS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("SECTION 1: Flight Science / Aero Parameters")
    print("=" * 50)

    C_m = get_input("C_m [-]", DEFAULTS['C_m'])
    AR = get_input("AR [-]", DEFAULTS['AR'])
    lambda_taper = get_input("λ (taper ratio) [-]", DEFAULTS['lambda_taper'])
    V_c = get_input("V_c [m/s]", DEFAULTS['V_c'])
    n = get_input("n (load factor) [-]", DEFAULTS['n'])
    S_ref = get_input("S_ref [m²]", DEFAULTS['S_ref'])
    W_0 = get_input("W_0 [N]", DEFAULTS['W_0'])
    rho_air = get_input("ρ_air [kg/m³]", DEFAULTS['rho_air'])
    x_ac_percent = get_input("x_ac [% chord]", DEFAULTS['x_ac_percent'])
    Lambda_ac_deg = get_input("Λ_ac [deg]", DEFAULTS['Lambda_ac_deg'])

    # ==========================================================================
    # SECTION 2: GEOMETRY
    # ==========================================================================
    print("\n" + "=" * 50)
    print("SECTION 2: Geometry Parameters")
    print("=" * 50)

    b = get_input("b (wingspan) [m]", DEFAULTS['b'])
    t_over_c = get_input("t/c [-]", DEFAULTS['t_over_c'])

    # Calculate C_r from formula: C_r = 2 * S_ref / (b * (1 + λ))
    C_r = 2 * S_ref / (b * (1 + lambda_taper))  # [m]
    C_r_mm = C_r * 1000  # [mm]

    # Calculate c_MGC from formula: c_MGC = (4/3) * sqrt(S_ref/AR) * ((1 + λ + λ²) / (1 + 2λ + λ²))
    c_MGC = (4 / 3) * np.sqrt(S_ref / AR) * ((1 + lambda_taper + lambda_taper**2) / (1 + 2*lambda_taper + lambda_taper**2))

    # Calculate Ȳ from formula: Ȳ = (b/6) * ((1 + 2λ) / (1 + λ)) * 1000
    Y_bar_mm = (b / 6) * ((1 + 2 * lambda_taper) / (1 + lambda_taper)) * 1000

    # Calculate N_Rib from formula: N_Rib = ceil(1 + sqrt(AR * S_ref) / c_MGC)
    N_Rib = int(np.ceil(1 + np.sqrt(AR * S_ref) / c_MGC))

    # Calculate additional geometry values
    L_span = b / 2  # Half-wing span [m]
    c_tip = lambda_taper * C_r  # Tip chord [m]
    h_box_root = t_over_c * C_r  # Wing box height at root [m]

    print(f"\n  Calculated geometry parameters:")
    print(f"    L_span (half-wing span) = {L_span*1000:.2f} mm ({L_span:.4f} m)")
    print(f"    C_r (root chord) = {C_r_mm:.2f} mm")
    print(f"    c_tip (tip chord) = {c_tip*1000:.2f} mm")
    print(f"    c_MGC (mean geometric chord) = {c_MGC*1000:.2f} mm ({c_MGC:.4f} m)")
    print(f"    Ȳ (MGC spanwise position) = {Y_bar_mm:.2f} mm")
    print(f"    h_box (wing box height at root) = {h_box_root*1000:.2f} mm")
    print(f"    N_Rib (number of ribs) = {N_Rib}")

    # ==========================================================================
    # SECTION 3: MATERIALS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("SECTION 3: Material Selection")
    print("=" * 50)

    print_material_table(db)
    mat_spar_key = select_material(db, "Spar material ID", DEFAULTS['material_spar_id'])
    mat_rib_key = select_material(db, "Rib material ID", DEFAULTS['material_rib_id'])
    mat_skin_key = select_material(db, "Skin material ID", DEFAULTS['material_skin_id'])

    materials = MaterialSelection.from_database(db, mat_spar_key, mat_skin_key, mat_rib_key)

    print(f"\nSelected materials:")
    print(f"  Spar: {materials.spar.name}")
    print(f"  Skin: {materials.skin.name}")
    print(f"  Rib:  {materials.rib.name}")

    # ==========================================================================
    # SECTION 4: STRUCTURAL PARAMETERS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("SECTION 4: Structural Parameters")
    print("=" * 50)

    t_skin_mm = get_input("t_skin [mm]", DEFAULTS['t_skin_mm'])
    SF = get_input("SF (safety factor) [-]", DEFAULTS['SF'])
    theta_max_deg = get_input("θ_max (max tip twist) [deg]", DEFAULTS['theta_max_deg'])
    s_min_mm = get_input("s_min (min rib spacing) [mm]", DEFAULTS['s_min_mm'])
    N_Rib_max_factor = get_input("N_Rib_max factor (N_Rib_max = factor × N_Rib_min)", DEFAULTS['N_Rib_max_factor'])
    t_skin_step_mm = get_input("t_skin step (Phase-2 skin increment) [mm]", DEFAULTS['t_skin_step_mm'])

    # Rib profile type selection
    print(f"\n  Rib profile type:")
    print(f"    1 = Rectangular (conservative)")
    print(f"    2 = Parabolic (airfoil approximation)")
    rib_profile = get_input("  Select rib profile (1 or 2)", DEFAULTS['rib_profile'], int)
    rib_profile_name = "Parabolic" if rib_profile == 2 else "Rectangular"

    # Load distribution selection
    print(f"\n  Lift distribution type:")
    print(f"    1 = Uniform")
    print(f"    2 = Elliptic")
    load_dist_type = get_input("  Select lift distribution (1 or 2)", DEFAULTS['load_dist_type'], int)
    pitch_dist_type = load_dist_type  # Match pitch distribution to lift

    # Buckling mode selection
    print(f"\n  Buckling check mode:")
    print(f"    1 = Shear buckling only")
    print(f"    2 = Shear + Compression buckling")
    buckling_mode = get_input("  Select buckling mode (1 or 2)", DEFAULTS['buckling_mode'], int)

    load_dist_name = "Elliptic" if load_dist_type == 2 else "Uniform"
    buckling_mode_name = "Shear + Compression" if buckling_mode == 2 else "Shear only"

    print(f"\n  Selected parameters:")
    print(f"    Rib profile = {rib_profile_name}")
    print(f"    Lift distribution = {load_dist_name}")
    print(f"    Buckling mode = {buckling_mode_name}")
    print(f"    N_Rib_max factor = {N_Rib_max_factor:.1f}x (N_Rib_max = {N_Rib_max_factor:.1f} × N_Rib_min)")
    print(f"    t_skin step = {t_skin_step_mm:.1f} mm (Phase-2 skin increment)")

    # ==========================================================================
    # SECTION 5: DESIGN RANGES
    # ==========================================================================
    print("\n" + "=" * 50)
    print("SECTION 5: Design Ranges for Optimization")
    print("=" * 50)

    # N_Rib is calculated, create single-value range
    N_Rib_range = DesignRange(min_val=N_Rib, max_val=N_Rib, step=1)

    print("\nRib Parameters:")
    t_rib_range = get_range_input("t_rib", "mm", 1.0, 5.0, 1.0)

    print("\nFront Spar Parameters:")
    X_FS_range = get_range_input("X_FS%", "%", 15.0, 30.0, 5.0)
    d_FS_range = get_range_input("d_FS_outer", "mm", 16.0, 30.0, 2.0)
    t_FS_range = get_range_input("t_FS", "mm", 0.5, 3.0, 0.5)

    print("\nRear Spar Parameters:")
    X_RS_range = get_range_input("X_RS%", "%", 40.0, 65.0, 5.0)
    d_RS_range = get_range_input("d_RS_outer", "mm", 10.0, 25.0, 5.0)
    t_RS_range = get_range_input("t_RS", "mm", 0.5, 3.0, 0.5)

    design_space = DesignSpace(
        N_Rib=N_Rib_range,
        t_rib_mm=t_rib_range,
        X_FS_percent=X_FS_range,
        X_RS_percent=X_RS_range,
        d_FS_outer_mm=d_FS_range,
        t_FS_mm=t_FS_range,
        d_RS_outer_mm=d_RS_range,
        t_RS_mm=t_RS_range,
    )

    print(f"\nTotal combinations (grid search): {design_space.total_combinations}")

    # ==========================================================================
    # SECTION 6: OPTIMIZATION METHOD SELECTION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("SECTION 6: Optimization Method")
    print("=" * 50)

    print("\n  1 = Grid Search (exhaustive, evaluates all combinations)")
    print(f"      → {design_space.total_combinations} evaluations")
    print("  2 = Genetic Algorithm (faster, explores continuous space)")
    print("      → ~15,000 evaluations (default)")
    opt_method = get_input("  Select method (1 or 2)", 1, int)

    n_workers = 1
    ga_config = None
    if opt_method == 2:
        pop_size = get_input("  GA population size", 100, int)
        n_gen = get_input("  GA generations", 150, int)
        ga_config = GAConfig(pop_size=pop_size, n_gen=n_gen)
        print(f"  → ~{pop_size * n_gen} evaluations")
    else:
        n_workers = get_input("  Parallel workers (CPU cores)", 10, int)
        if n_workers < 1:
            n_workers = 1

    # ==========================================================================
    # RUN OPTIMIZATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("RUNNING OPTIMIZATION")
    print("=" * 50)

    # Create objects
    planform = PlanformParams.from_input(
        b=b, AR=AR, taper_ratio=lambda_taper, t_c=t_over_c,
        S_ref=S_ref, C_r_mm=C_r_mm, c_MGC=c_MGC, Y_bar_mm=Y_bar_mm
    )

    flight = FlightCondition(
        W0=W_0, n=n, V_c=V_c, rho=rho_air,
        C_m=C_m, S_ref=S_ref, c_MGC=c_MGC
    )

    aero_center = AeroCenter(x_ac_percent=x_ac_percent, Lambda_ac_deg=Lambda_ac_deg)

    load_dist = "elliptic" if load_dist_type == 2 else "uniform"
    pitch_dist = "chord_weighted" if pitch_dist_type == 2 else "uniform"

    # Calculate wing box height at root for assembly check (h_FS = h_RS = t/c * C_r)
    h_box_root_mm = t_over_c * C_r_mm  # [mm]

    use_parabolic = (rib_profile == 2)

    if opt_method == 2:
        best, optimizer = run_ga_optimization(
            planform, flight, aero_center, materials, design_space,
            t_skin_mm=t_skin_mm, SF=SF,
            h_FS_mm=h_box_root_mm, h_RS_mm=h_box_root_mm,
            theta_max_deg=theta_max_deg,
            rib_parabolic=use_parabolic,
            load_dist=load_dist, pitch_dist=pitch_dist,
            ga_config=ga_config,
            s_min_mm=s_min_mm,
            buckling_mode=buckling_mode,
            N_Rib_max_factor=N_Rib_max_factor,
            t_skin_step_mm=t_skin_step_mm
        )
    else:
        best, optimizer = run_optimization(
            planform, flight, aero_center, materials, design_space,
            t_skin_mm=t_skin_mm, SF=SF,
            h_FS_mm=h_box_root_mm, h_RS_mm=h_box_root_mm,
            theta_max_deg=theta_max_deg,
            rib_parabolic=use_parabolic,
            load_dist=load_dist, pitch_dist=pitch_dist,
            n_workers=n_workers,
            s_min_mm=s_min_mm,
            buckling_mode=buckling_mode,
            N_Rib_max_factor=N_Rib_max_factor,
            t_skin_step_mm=t_skin_step_mm
        )

    if best is None:
        print("\nERROR: No valid solution found!")
        print("Try relaxing constraints or expanding design space.")
        return

    # ==========================================================================
    # GENERATE DETAILED RESULTS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("GENERATING DETAILED RESULTS")
    print("=" * 50)

    # Recompute with optimal configuration for detailed outputs
    cfg = best.config
    spar_pos = SparPosition(X_FS_percent=cfg.X_FS_percent, X_RS_percent=cfg.X_RS_percent)
    stations = compute_all_stations(planform, spar_pos, cfg.N_Rib)

    y = np.array([s.y for s in stations])
    chord = np.array([s.chord for s in stations])
    x_FS = np.array([s.x_FS for s in stations])
    x_RS = np.array([s.x_RS for s in stations])
    A_m = np.array([s.A_m for s in stations])
    h_box = np.array([s.h_box for s in stations])
    box_width = np.array([s.width for s in stations])

    # Load analysis
    ld_type = LoadDistributionType.ELLIPTIC if load_dist_type == 2 else LoadDistributionType.UNIFORM
    pd_type = PitchMomentDistributionType.CHORD_WEIGHTED if pitch_dist_type == 2 else PitchMomentDistributionType.UNIFORM

    loads = analyze_loads(y, chord, x_FS, x_RS, flight, aero_center, planform.L_span, ld_type, pd_type)
    reactions = compute_root_reactions(loads)

    # Torsion (use Phase-2 final t_skin if it was adjusted)
    t_skin_final_mm = best.t_skin_final_mm if best.t_skin_final_mm > 0 else t_skin_mm
    t_skin = t_skin_final_mm / 1000
    torsion_results = analyze_torsion_stations(y, loads.T, A_m, t_skin, materials.skin.G, box_width, h_box)

    # Spars
    spar_FS = SparProperties.from_mm(cfg.d_FS_outer_mm, cfg.t_FS_mm, materials.spar)
    spar_RS = SparProperties.from_mm(cfg.d_RS_outer_mm, cfg.t_RS_mm, materials.spar)
    spar_results = analyze_spars_all_stations(
        y, loads.M, loads.T, spar_FS, spar_RS,
        cfg.X_FS_percent, cfg.X_RS_percent, x_ac_percent
    )

    # Ribs
    rib_props = RibProperties.from_mm(cfg.t_rib_mm, materials.rib)
    ribs = generate_rib_geometries(y, chord, h_box, x_FS, x_RS, parabolic=use_parabolic)
    rib_outputs = format_rib_outputs(ribs, rib_props, planform.L_span)

    # Box beam analysis
    E_skin = materials.skin.E
    box_sections = [
        compute_box_beam_section(spar_FS, spar_RS, h_box[i], box_width[i], t_skin, E_skin)
        for i in range(len(y))
    ]

    sharing = LoadSharing.from_positions(cfg.X_FS_percent, cfg.X_RS_percent, x_ac_percent)

    box_results = [
        analyze_box_beam_stress(loads.M[i], loads.T[i], box_sections[i],
                                spar_FS, spar_RS, sharing.eta_FS, sharing.eta_RS, y[i])
        for i in range(len(y))
    ]

    # Bay-level buckling analysis
    allowables_dict = optimizer.compute_allowables()
    tau_skin_arr = np.array([abs(tr.tau_skin) for tr in torsion_results])
    sigma_comp_arr = np.array([br.sigma_skin_comp for br in box_results])

    tau_interp = interp1d(y, tau_skin_arr, kind='linear', fill_value='extrapolate')
    sigma_interp = interp1d(y, sigma_comp_arr, kind='linear', fill_value='extrapolate')
    V_interp = interp1d(y, loads.V, kind='linear', fill_value='extrapolate')
    h_interp = interp1d(y, h_box, kind='linear', fill_value='extrapolate')
    M_interp = interp1d(y, loads.M, kind='linear', fill_value='extrapolate')
    T_interp = interp1d(y, loads.T, kind='linear', fill_value='extrapolate')
    chord_interp = interp1d(y, chord, kind='linear', fill_value='extrapolate')
    A_m_interp = interp1d(y, A_m, kind='linear', fill_value='extrapolate')
    vm_FS_interp = interp1d(y, np.array([br.sigma_vm_FS for br in box_results]),
                             kind='linear', fill_value='extrapolate')
    vm_RS_interp = interp1d(y, np.array([br.sigma_vm_RS for br in box_results]),
                             kind='linear', fill_value='extrapolate')

    adapt_config = AdaptiveRibConfig(
        E_skin=materials.skin.E,
        nu_skin=materials.skin.nu,
        t_skin=t_skin,
        E_rib=materials.rib.E,
        nu_rib=materials.rib.nu,
        t_rib=rib_props.t_rib,
        tau_allow_skin=allowables_dict['tau_allow_skin'],
        sigma_allow_skin=allowables_dict['sigma_allow_skin'],
        buckling_mode=buckling_mode,
    )

    # Use Phase-2 final rib positions if available, otherwise recompute
    if best.final_y_ribs is not None and len(best.final_y_ribs) > 0:
        final_y_ribs = best.final_y_ribs
        rib_feasible = best.rib_feasible
        adapt_msg = f"Using Phase-2 rib layout: {len(final_y_ribs)} stations ({len(final_y_ribs)-1} bays)"
    else:
        final_y_ribs, rib_feasible, adapt_msg = adaptive_rib_insertion(
            y, adapt_config, tau_interp, sigma_interp, V_interp, h_interp
        )

    bay_results = compute_bay_results(
        final_y_ribs, adapt_config,
        tau_skin_at_y=tau_interp,
        sigma_comp_at_y=sigma_interp,
        V_at_y=V_interp,
        M_at_y=M_interp,
        T_at_y=T_interp,
        chord_at_y=chord_interp,
        h_box_at_y=h_interp,
        A_m_at_y=A_m_interp,
        sigma_vm_FS_at_y=vm_FS_interp,
        sigma_vm_RS_at_y=vm_RS_interp,
    )

    # Skin arc lengths
    skin_arcs = compute_skin_arc_lengths_root(planform, spar_pos)

    # Sweep angles (using aerodynamic formula)
    Lambda_FS = compute_spar_sweep_angle(cfg.X_FS_percent, planform, Lambda_ac_deg, x_ac_percent)
    Lambda_RS = compute_spar_sweep_angle(cfg.X_RS_percent, planform, Lambda_ac_deg, x_ac_percent)

    # Critical area using formula: A_Cri = (n × W_0 × Ȳ × 0.5 × η) / (2 × σ_max × c)
    # where c = d_outer/2 (distance to outer fiber)
    sigma_max_spar = materials.spar.sigma_u
    c_FS = spar_FS.d_outer / 2  # [m]
    c_RS = spar_RS.d_outer / 2  # [m]
    A_cri_FS = (n * W_0 * Y_bar_mm/1000 * 0.5 * best.eta_FS) / (2 * sigma_max_spar * c_FS)
    A_cri_RS = (n * W_0 * Y_bar_mm/1000 * 0.5 * best.eta_RS) / (2 * sigma_max_spar * c_RS)

    # ==========================================================================
    # BUILD REPORT OBJECTS
    # ==========================================================================

    S_skin_half = compute_skin_area_half_wing(planform)

    # Common spar/geometry fields shared by both Phase-1 and Phase-2 configs
    _common_config = dict(
        X_FS_percent=cfg.X_FS_percent,
        X_RS_percent=cfg.X_RS_percent,
        d_FS_outer_mm=cfg.d_FS_outer_mm,
        t_FS_mm=cfg.t_FS_mm,
        d_RS_outer_mm=cfg.d_RS_outer_mm,
        t_RS_mm=cfg.t_RS_mm,
        Lambda_FS_deg=Lambda_FS,
        Lambda_RS_deg=Lambda_RS,
        eta_FS_percent=best.eta_FS * 100,
        eta_RS_percent=best.eta_RS * 100,
        X_FS_mm=spar_pos.x_FS_at_station(planform.C_r) * 1000,
        X_RS_mm=spar_pos.x_RS_at_station(planform.C_r) * 1000,
        L_FS_mm=planform.L_span * 1000 / np.cos(np.radians(Lambda_FS)),
        L_RS_mm=planform.L_span * 1000 / np.cos(np.radians(Lambda_RS)),
        A_Act_FS_mm2=spar_FS.area * 1e6,
        A_Act_RS_mm2=spar_RS.area * 1e6,
        A_Cri_FS_mm2=A_cri_FS * 1e6,
        A_Cri_RS_mm2=A_cri_RS * 1e6,
        I_FS_mm4=spar_FS.I * 1e12,
        I_RS_mm4=spar_RS.I * 1e12,
        S_skin_m2=S_skin_half,
        L_Skin_LE_FS_mm=skin_arcs.L_LE_FS * 1000,
        L_Skin_LE_RS_mm=skin_arcs.L_LE_RS * 1000,
        L_Skin_FS_RS_mm=skin_arcs.L_FS_RS * 1000,
        S_Rib_LE_FS_mm2=rib_outputs.S_Rib_LE_FS_mm2,
        S_Rib_FS_RS_mm2=rib_outputs.S_Rib_FS_RS_mm2,
        S_Rib_mm2=rib_outputs.S_Rib_total_mm2,
    )

    # --- PHASE-1 config (original N_Rib, user-input t_skin) ---
    phase1_config = OptimalConfig(
        N_Rib=cfg.N_Rib,
        t_rib_mm=cfg.t_rib_mm,
        rib_spacing_mm=planform.L_span / cfg.N_Rib * 1000,
        t_skin_mm=t_skin_mm,
        **_common_config,
    )

    phase1_mass = MassBreakdown(
        m_skin=best.mass_skin_p1 if best.mass_skin_p1 > 0 else best.mass_skin,
        m_FS=best.mass_FS,
        m_RS=best.mass_RS,
        m_ribs=best.mass_ribs_p1 if best.mass_ribs_p1 > 0 else best.mass_ribs,
        m_total=best.mass_total_p1 if best.mass_total_p1 > 0 else best.mass_total,
    )

    # --- PHASE-2 config (final N_Rib, final t_skin after buckling fix) ---
    N_Rib_final = best.N_Rib_final if best.N_Rib_final > 0 else cfg.N_Rib
    rib_spacing_avg = planform.L_span / N_Rib_final * 1000 if N_Rib_final > 0 else 0.0

    phase2_config = OptimalConfig(
        N_Rib=N_Rib_final,
        t_rib_mm=cfg.t_rib_mm,
        rib_spacing_mm=rib_spacing_avg,
        t_skin_mm=t_skin_final_mm,
        **_common_config,
    )

    mass_breakdown = MassBreakdown(
        m_skin=best.mass_skin,
        m_FS=best.mass_FS,
        m_RS=best.mass_RS,
        m_ribs=best.mass_ribs,
        m_total=best.mass_total
    )

    allowables = optimizer.compute_allowables()
    tau_allow = allowables['tau_allow_skin']
    sigma_allow = allowables['sigma_allow_spar']
    tau_allow_spar = allowables['tau_allow_spar']

    # ==========================================================================
    # FINAL VALIDATION - Ensure best solution passes ALL acceptance criteria
    # ==========================================================================
    h_box_mm = h_box_root_mm  # Wing box height for assembly check

    def validate_solution(sol):
        """Validate that solution passes ALL 8 acceptance criteria."""
        errors = []
        cfg = sol.config

        # 1. Skin shear stress: τ_skin < τ_allow
        if sol.tau_skin_max > tau_allow:
            errors.append(f"Skin stress: {sol.tau_skin_max/1e6:.2f} > {tau_allow/1e6:.2f} MPa")

        # 2. Front spar interaction: (σ/σ_allow) + (τ/τ_allow)² < 1
        FS_interaction = (sol.sigma_b_FS_max / sigma_allow) + (sol.tau_FS_max / tau_allow_spar) ** 2
        if FS_interaction >= 1.0:
            errors.append(f"FS interaction: {FS_interaction:.3f} >= 1.0")

        # 3. Rear spar interaction: (σ/σ_allow) + (τ/τ_allow)² < 1
        RS_interaction = (sol.sigma_b_RS_max / sigma_allow) + (sol.tau_RS_max / tau_allow_spar) ** 2
        if RS_interaction >= 1.0:
            errors.append(f"RS interaction: {RS_interaction:.3f} >= 1.0")

        # 4. Front spar area: A_Act_FS > A_Cri_FS
        if sol.A_Act_FS <= sol.A_Cri_FS:
            errors.append(f"FS area: {sol.A_Act_FS*1e6:.2f} <= {sol.A_Cri_FS*1e6:.2f} mm²")

        # 5. Rear spar area: A_Act_RS > A_Cri_RS
        if sol.A_Act_RS <= sol.A_Cri_RS:
            errors.append(f"RS area: {sol.A_Act_RS*1e6:.2f} <= {sol.A_Cri_RS*1e6:.2f} mm²")

        # 6. Front spar assembly: d_FS + 10 < h_box
        if cfg.d_FS_outer_mm + 10 >= h_box_mm:
            errors.append(f"FS assembly: {cfg.d_FS_outer_mm}+10 >= {h_box_mm:.1f} mm")

        # 7. Rear spar assembly: d_RS + 10 < h_box
        if cfg.d_RS_outer_mm + 10 >= h_box_mm:
            errors.append(f"RS assembly: {cfg.d_RS_outer_mm}+10 >= {h_box_mm:.1f} mm")

        # 8. Tip twist: θ_tip ≤ θ_max
        if sol.theta_tip_deg > theta_max_deg:
            errors.append(f"Twist: {sol.theta_tip_deg:.2f}° > {theta_max_deg:.1f}°")

        # 9. Skin compression buckling (only in shear+compression mode)
        # NOTE: Global sigma_skin_comp_max vs sigma_cr_min comparison disabled.
        # This was comparing root compression stress against tip buckling critical
        # (different bays), giving false failures. Phase-2 already performs correct
        # bay-by-bay buckling checks. Uncomment below to re-enable if needed.
        if buckling_mode == 2:
            # if hasattr(sol, 'sigma_cr_min') and sol.sigma_cr_min > 0:
            #     if sol.sigma_skin_comp_max > sol.sigma_cr_min:
            #         errors.append(f"Compression buckling: sigma_skin={sol.sigma_skin_comp_max/1e6:.2f} > sigma_cr={sol.sigma_cr_min/1e6:.2f} MPa")

            # 10. Combined interaction
            if hasattr(sol, 'R_combined_max'):
                if sol.R_combined_max >= 1.0:
                    errors.append(f"Combined interaction: R={sol.R_combined_max:.3f} >= 1.0")

        return len(errors) == 0, errors

    # Iterate through solutions until we find one that passes validation
    valid_best = None
    all_solutions = optimizer.get_top_solutions(len(optimizer.accepted_results))

    for candidate in all_solutions:
        is_valid, errors = validate_solution(candidate)
        if is_valid:
            valid_best = candidate
            break
        else:
            print(f"  Solution rejected (mass={candidate.mass_total*1000:.1f}g): {', '.join(errors)}")

    if valid_best is None:
        print("\nERROR: No solution passes final validation!")
        print("All accepted solutions fail at least one criterion.")
        print("Try expanding design space (larger diameters, thicker walls).")
        return

    # Update best to the validated solution
    if valid_best != best:
        print(f"\n  Original best ({best.mass_total*1000:.1f}g) failed validation.")
        print(f"  Using validated best: {valid_best.mass_total*1000:.1f}g")
        best = valid_best
        cfg = best.config
        # Recalculate spar properties with new config
        spar_FS = SparProperties.from_mm(cfg.d_FS_outer_mm, cfg.t_FS_mm, materials.spar)
        spar_RS = SparProperties.from_mm(cfg.d_RS_outer_mm, cfg.t_RS_mm, materials.spar)
        spar_pos = SparPosition(X_FS_percent=cfg.X_FS_percent, X_RS_percent=cfg.X_RS_percent)

    stress_results = StressResults(
        tau_skin_max_MPa=best.tau_skin_max / 1e6,
        tau_skin_allow_MPa=tau_allow / 1e6,
        tau_skin_margin_percent=(tau_allow - best.tau_skin_max) / tau_allow * 100,
        sigma_b_FS_max_MPa=best.sigma_b_FS_max / 1e6,
        tau_FS_max_MPa=best.tau_FS_max / 1e6,
        sigma_vm_FS_max_MPa=best.sigma_vm_FS_max / 1e6,
        sigma_FS_allow_MPa=sigma_allow / 1e6,
        sigma_FS_margin_percent=(sigma_allow - best.sigma_vm_FS_max) / sigma_allow * 100,
        sigma_b_RS_max_MPa=best.sigma_b_RS_max / 1e6,
        tau_RS_max_MPa=best.tau_RS_max / 1e6,
        sigma_vm_RS_max_MPa=best.sigma_vm_RS_max / 1e6,
        sigma_RS_allow_MPa=sigma_allow / 1e6,
        sigma_RS_margin_percent=(sigma_allow - best.sigma_vm_RS_max) / sigma_allow * 100,
        y_crit_mm=0.0,  # Root is typically critical
        critical_component="Front Spar" if best.sigma_vm_FS_max > best.sigma_vm_RS_max else "Rear Spar",
        theta_tip_deg=best.theta_tip_deg,
        theta_max_deg=theta_max_deg,
        # Formula components for display
        M_root_Nm=reactions.Mx,  # Bending moment at root
        d_FS_mm=cfg.d_FS_outer_mm,
        t_FS_mm=cfg.t_FS_mm,
        d_RS_mm=cfg.d_RS_outer_mm,
        t_RS_mm=cfg.t_RS_mm,
        eta_FS=best.eta_FS,
        eta_RS=best.eta_RS
    )

    root_reactions = RootReactionsReport(
        Fx_N=reactions.Fx,
        Fy_N=reactions.Fy,
        Fz_N=reactions.Fz,
        Mx_Nm=reactions.Mx,
        My_Nm=reactions.My,
        Mz_Nm=reactions.Mz
    )

    mat_names = {
        "Spar": materials.spar.name,
        "Skin": materials.skin.name,
        "Rib": materials.rib.name
    }

    top_solutions = []
    for sol in optimizer.get_top_solutions(10):
        top_solutions.append({
            'mass_g': sol.mass_total * 1000,
            'N_Rib': sol.config.N_Rib,
            'N_Rib_final': sol.N_Rib_final,
            'X_FS': sol.config.X_FS_percent,
            'X_RS': sol.config.X_RS_percent,
            'd_FS_mm': sol.config.d_FS_outer_mm,
            't_FS_mm': sol.config.t_FS_mm,
            'd_RS_mm': sol.config.d_RS_outer_mm,
            't_RS_mm': sol.config.t_RS_mm,
            'R_max': sol.R_combined_max,
        })

    opt_history = OptimizationHistory(
        total_combinations=design_space.total_combinations,
        valid_combinations=sum(1 for r in optimizer.results if r.valid),
        accepted_combinations=len(optimizer.accepted_results),
        rejection_reasons=optimizer.get_rejection_summary(),
        best_solutions=top_solutions
    )

    # ==========================================================================
    # PHASE-2 POST-PROCESSING: Rib web buckling fix & mass update
    # (Compute first, print later — so MASS BREAKDOWN shows final values)
    # ==========================================================================

    # --- Rib web buckling check & auto-thickening ---
    rwb = fix_rib_web_buckling(
        final_y_ribs,
        V_at_y=V_interp,
        h_box_at_y=h_interp,
        E_rib=materials.rib.E,
        nu_rib=materials.rib.nu,
        t_rib_initial=cfg.t_rib_mm / 1000,
        t_rib_step=0.0005,
        t_rib_max=0.010,
    )

    # Compute rib geometries at final stations for mass display
    from geometry import chord_at_station as chord_at_station_val
    rwb_chord = np.array([
        chord_at_station_val(yi, planform.C_r, planform.c_tip, planform.L_span)
        for yi in final_y_ribs
    ])
    rwb_h_box = planform.t_c * rwb_chord
    rwb_x_FS_arr = np.array([spar_pos.x_FS_at_station(c) for c in rwb_chord])
    rwb_x_RS_arr = np.array([spar_pos.x_RS_at_station(c) for c in rwb_chord])
    rwb_rib_geoms = generate_rib_geometries(
        final_y_ribs, rwb_chord, rwb_h_box, rwb_x_FS_arr, rwb_x_RS_arr,
        parabolic=use_parabolic)

    # Update mass if ribs were thickened
    old_rib_mass = best.mass_ribs
    if rwb.n_thickened > 0:
        total_rib_mass_new = 0.0
        for ri in range(1, len(rwb_rib_geoms)):
            t_ri = rwb.t_rib_per_station[ri]
            total_rib_mass_new += rwb_rib_geoms[ri].S_total * t_ri * materials.rib.density

        best.mass_ribs = total_rib_mass_new
        best.mass_total = best.mass_skin + best.mass_FS + best.mass_RS + best.mass_ribs
        mass_breakdown.m_ribs = best.mass_ribs
        mass_breakdown.m_total = best.mass_total

    # ==========================================================================
    # PRINT PHASE-1 REPORT (original config, original masses)
    # ==========================================================================
    report_p1 = generate_full_report(phase1_config, phase1_mass, stress_results,
                                     root_reactions, mat_names, opt_history,
                                     include_mass=True, include_history=False,
                                     phase_label="PHASE-1 (Initial Sizing)")
    print(report_p1)

    # ==========================================================================
    # PRINT PHASE-2 REPORT (final config after buckling fix, final masses)
    # ==========================================================================
    report_p2 = generate_full_report(phase2_config, mass_breakdown, stress_results,
                                     root_reactions, mat_names, opt_history,
                                     include_mass=True, include_history=False,
                                     phase_label="PHASE-2 (After Buckling Fix)")
    print(report_p2)

    # Warnings
    warnings = check_for_warnings(stress_results, mass_breakdown)
    if warnings:
        print("\n" + "!" * 70)
        for w in warnings:
            print(w)
        print("!" * 70)

    # ==========================================================================
    # PHASE-2 DETAIL TABLES
    # ==========================================================================

    # --- Box Beam & Buckling Summary ---
    print("\n" + "=" * 50)
    print("BOX BEAM & BUCKLING ANALYSIS")
    print("=" * 50)

    print(f"\n  Box Beam (Transformed Section):")
    print(f"    I_box (root)           = {best.I_box_root*1e12:.2f} mm4")
    print(f"    I_FS (spar)            = {best.I_FS*1e12:.2f} mm4")
    print(f"    I_RS (spar)            = {best.I_RS*1e12:.2f} mm4")
    print(f"    Skin I fraction        = {best.skin_fraction*100:.1f}%")

    print(f"\n  Buckling Summary:")
    print(f"    tau_cr_min (shear)       = {best.tau_cr_min/1e6:.4f} MPa")
    print(f"    sigma_cr_min (compression) = {best.sigma_cr_min/1e6:.4f} MPa")
    print(f"    sigma_skin_comp_max        = {best.sigma_skin_comp_max/1e6:.4f} MPa")
    print(f"    R_combined_max         = {best.R_combined_max:.4f}")
    print(f"    Rib feasible           = {best.rib_feasible}")
    print(f"    N_Rib initial          = {best.config.N_Rib}")
    print(f"    N_Rib final            = {best.N_Rib_final}")

    # Print adaptive rib status
    print(f"\n  {adapt_msg}")

    # --- Bay-by-Bay Buckling Results ---
    if bay_results:
        print("\n" + "=" * 50)
        print("BAY-BY-BAY BUCKLING RESULTS")
        print("=" * 50)
        print(format_bay_results_table(bay_results))

    # --- Rib Web Buckling Table ---
    print("\n" + "=" * 50)
    print("RIB WEB BUCKLING CHECK")
    print("=" * 50)

    print(format_rib_web_buckling_table(rwb, rib_geometries=rwb_rib_geoms,
                                         rib_density=materials.rib.density))

    if rwb.n_thickened > 0:
        print(f"\n  {rwb.n_thickened} ribs thickened: "
              f"{rwb.t_rib_initial_mm:.1f}mm -> {rwb.t_rib_max_mm:.1f}mm (max)")
        print(f"  Rib mass updated: {old_rib_mass*1000:.2f}g -> {best.mass_ribs*1000:.2f}g")
        print(f"  Total mass (final): {best.mass_total*1000:.2f}g")
    else:
        print(f"\n  All ribs pass web buckling at t_rib = {rwb.t_rib_initial_mm:.1f}mm. No thickening needed.")

    # ==========================================================================
    # MASS BREAKDOWN (after all Phase-2 adjustments)
    # ==========================================================================
    print("\n" + generate_mass_table(mass_breakdown))

    # ==========================================================================
    # OPTIMIZATION HISTORY
    # ==========================================================================
    if opt_history:
        print("\n" + generate_optimization_history_section(opt_history))

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)

    # ==========================================================================
    # GENERATE PLOTS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("GENERATING PLOTS")
    print("=" * 50)

    plot_config = PlotConfig(
        save_plots=True,
        show_plots=False,
        output_dir="output_plots"
    )

    # Extract data for plots
    tau_skin = np.array([r.tau_skin for r in torsion_results])
    q = np.array([r.q for r in torsion_results])
    twist_rate = np.array([r.twist_rate for r in torsion_results])
    sigma_vm_FS = np.array([r.FS.sigma_vm for r in spar_results])
    sigma_vm_RS = np.array([r.RS.sigma_vm for r in spar_results])

    # Build Phase-2 data for planform comparison plot
    phase2_plot_data = None
    if final_y_ribs is not None and len(final_y_ribs) > 2:
        phase2_plot_data = {
            'y_phase2': final_y_ribs,
            'chord_at_y': chord_interp,
            'x_FS_at_y': lambda yi: np.interp(yi, y, x_FS),
            'x_RS_at_y': lambda yi: np.interp(yi, y, x_RS),
        }

    generate_all_plots(
        y=y,
        loads_data={'w': loads.w, 'V': loads.V, 'M': loads.M},
        torsion_data={'T': loads.T, 'q': q, 'tau_skin': tau_skin, 'twist_rate': twist_rate},
        spar_data={'sigma_vm_FS': sigma_vm_FS, 'sigma_vm_RS': sigma_vm_RS},
        geometry_data={'chord': chord, 'x_FS': x_FS, 'x_RS': x_RS},
        allowables={'tau_allow_skin': tau_allow, 'sigma_allow_spar': sigma_allow},
        config=plot_config,
        phase2_data=phase2_plot_data,
    )

    # ==========================================================================
    # EXPORT RESULTS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("EXPORTING RESULTS")
    print("=" * 50)

    from dataclasses import asdict
    export_data = {
        'optimal_config': asdict(phase2_config),
        'mass_breakdown': asdict(mass_breakdown),
        'stress_results': asdict(stress_results),
        'root_reactions': asdict(root_reactions),
        'materials': mat_names,
        'box_beam': {
            'I_box_root_mm4': best.I_box_root * 1e12,
            'I_FS_mm4': best.I_FS * 1e12,
            'I_RS_mm4': best.I_RS * 1e12,
            'skin_fraction': best.skin_fraction,
        },
        'buckling': {
            'tau_cr_min_MPa': best.tau_cr_min / 1e6,
            'sigma_cr_min_MPa': best.sigma_cr_min / 1e6,
            'sigma_skin_comp_max_MPa': best.sigma_skin_comp_max / 1e6,
            'R_combined_max': best.R_combined_max,
            'N_Rib_final': best.N_Rib_final,
            'rib_feasible': best.rib_feasible,
        },
        'optimization_history': {
            'total': opt_history.total_combinations,
            'accepted': opt_history.accepted_combinations,
            'rejection_reasons': opt_history.rejection_reasons
        }
    }
    export_to_json(export_data, "optimization_results.json")

    # Export bay results CSV
    if bay_results:
        bay_csv = format_bay_results_csv(bay_results)
        with open("bay_results.csv", 'w') as f:
            f.write(bay_csv)
        print(f"  Bay results saved to: bay_results.csv")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"  Results saved to: optimization_results.json")
    print(f"  Plots saved to: output_plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
reporting.py - Results Reporting Module

Generates text reports, tables, and exports results to CSV/JSON.
"""

import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class MassBreakdown:
    """Mass breakdown for half-wing components."""
    m_skin: float       # Skin mass [kg]
    m_FS: float         # Front spar mass [kg]
    m_RS: float         # Rear spar mass [kg]
    m_ribs: float       # Total rib mass [kg]
    m_total: float      # Total half-wing mass [kg]

    def to_dict_grams(self) -> Dict[str, float]:
        """Convert to dictionary with values in grams."""
        return {
            'Skin': self.m_skin * 1000,
            'Front Spar': self.m_FS * 1000,
            'Rear Spar': self.m_RS * 1000,
            'Ribs': self.m_ribs * 1000,
            'Total': self.m_total * 1000
        }


@dataclass
class OptimalConfig:
    """Optimal configuration parameters."""
    # Rib parameters
    N_Rib: int
    t_rib_mm: float
    rib_spacing_mm: float

    # Spar positions
    X_FS_percent: float
    X_RS_percent: float

    # Front spar geometry
    d_FS_outer_mm: float
    t_FS_mm: float

    # Rear spar geometry
    d_RS_outer_mm: float
    t_RS_mm: float

    # Computed outputs
    Lambda_FS_deg: float    # Front spar sweep [deg]
    Lambda_RS_deg: float    # Rear spar sweep [deg]
    eta_FS_percent: float   # Front spar load share [%]
    eta_RS_percent: float   # Rear spar load share [%]
    X_FS_mm: float          # Front spar position at root [mm]
    X_RS_mm: float          # Rear spar position at root [mm]
    L_FS_mm: float          # Front spar length [mm]
    L_RS_mm: float          # Rear spar length [mm]
    A_Act_FS_mm2: float     # Front spar actual area [mm²]
    A_Act_RS_mm2: float     # Rear spar actual area [mm²]
    A_Cri_FS_mm2: float     # Front spar critical area [mm²]
    A_Cri_RS_mm2: float     # Rear spar critical area [mm²]
    I_FS_mm4: float         # Front spar inertia [mm⁴]
    I_RS_mm4: float         # Rear spar inertia [mm⁴]

    # Skin arc lengths at root
    L_Skin_LE_FS_mm: float
    L_Skin_LE_RS_mm: float
    L_Skin_FS_RS_mm: float

    # Rib areas at root
    S_Rib_LE_FS_mm2: float
    S_Rib_FS_RS_mm2: float
    S_Rib_mm2: float

    # Skin parameters (defaults so they are optional for backward compat)
    t_skin_mm: float = 0.0          # Skin thickness [mm]
    S_skin_m2: float = 0.0          # Skin area (half-wing) [m²]


@dataclass
class StressResults:
    """Critical stress results."""
    # Skin
    tau_skin_max_MPa: float
    tau_skin_allow_MPa: float
    tau_skin_margin_percent: float

    # Front spar
    sigma_b_FS_max_MPa: float    # Bending stress
    tau_FS_max_MPa: float        # Shear stress
    sigma_vm_FS_max_MPa: float   # Von Mises stress
    sigma_FS_allow_MPa: float
    sigma_FS_margin_percent: float

    # Rear spar
    sigma_b_RS_max_MPa: float    # Bending stress
    tau_RS_max_MPa: float        # Shear stress
    sigma_vm_RS_max_MPa: float   # Von Mises stress
    sigma_RS_allow_MPa: float
    sigma_RS_margin_percent: float

    # Critical station
    y_crit_mm: float
    critical_component: str

    # Tip twist
    theta_tip_deg: float = 0.0
    theta_max_deg: float = 2.0

    # Formula components for display
    M_root_Nm: float = 0.0
    d_FS_mm: float = 0.0
    t_FS_mm: float = 0.0
    d_RS_mm: float = 0.0
    t_RS_mm: float = 0.0
    eta_FS: float = 0.0
    eta_RS: float = 0.0


@dataclass
class RootReactionsReport:
    """Root reaction forces and moments."""
    Fx_N: float
    Fy_N: float
    Fz_N: float
    Mx_Nm: float
    My_Nm: float
    Mz_Nm: float


@dataclass
class OptimizationHistory:
    """Optimization process summary."""
    total_combinations: int
    valid_combinations: int
    accepted_combinations: int
    rejection_reasons: Dict[str, int]
    best_solutions: List[Dict[str, Any]]


# =============================================================================
# TEXT REPORT GENERATION
# =============================================================================

def generate_header(phase_label: str = "") -> str:
    """Generate report header.

    Args:
        phase_label: Optional label like "PHASE-1" or "PHASE-2 (Final)" to
                     append to the title line.
    """
    title = "HALF-WING STRUCTURAL SIZING - OPTIMIZATION RESULTS"
    if phase_label:
        title += f"  [{phase_label}]"
    lines = [
        "=" * 70,
        title,
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 70,
    ]
    return "\n".join(lines)


def generate_optimal_params_section(config: OptimalConfig) -> str:
    """Generate optimal parameters section."""
    lines = [
        "",
        "OPTIMAL CONFIGURATION PARAMETERS",
        "-" * 40,
        "",
        "Rib Parameters:",
        f"  N_Rib                    = {config.N_Rib}",
        f"  t_rib                    = {config.t_rib_mm:.2f} mm",
        f"  Rib spacing (avg)        = {config.rib_spacing_mm:.2f} mm",
        "",
        "Spar Positions:",
        f"  X_FS                     = {config.X_FS_percent:.1f}% chord ({config.X_FS_mm:.2f} mm at root)",
        f"  X_RS                     = {config.X_RS_percent:.1f}% chord ({config.X_RS_mm:.2f} mm at root)",
        f"  Λ_FS (sweep)             = {config.Lambda_FS_deg:.2f}°",
        f"  Λ_RS (sweep)             = {config.Lambda_RS_deg:.2f}°",
        "",
        "Front Spar Geometry:",
        f"  d_FS_outer               = {config.d_FS_outer_mm:.2f} mm",
        f"  t_FS                     = {config.t_FS_mm:.2f} mm",
        f"  A_(Act-FS)               = {config.A_Act_FS_mm2:.2f} mm²",
        f"  A_(Cri-FS)               = {config.A_Cri_FS_mm2:.2f} mm²",
        f"  I_FS                     = {config.I_FS_mm4:.2f} mm⁴",
        f"  L_FS                     = {config.L_FS_mm:.2f} mm",
        "",
        "Rear Spar Geometry:",
        f"  d_RS_outer               = {config.d_RS_outer_mm:.2f} mm",
        f"  t_RS                     = {config.t_RS_mm:.2f} mm",
        f"  A_(Act-RS)               = {config.A_Act_RS_mm2:.2f} mm²",
        f"  A_(Cri-RS)               = {config.A_Cri_RS_mm2:.2f} mm²",
        f"  I_RS                     = {config.I_RS_mm4:.2f} mm⁴",
        f"  L_RS                     = {config.L_RS_mm:.2f} mm",
        "",
        "Load Sharing:",
        f"  η_FS                     = {config.eta_FS_percent:.1f}%",
        f"  η_RS                     = {config.eta_RS_percent:.1f}%",
        "",
        "Skin Parameters:",
        f"  t_skin                   = {config.t_skin_mm:.2f} mm",
        f"  S_skin (half-wing)       = {config.S_skin_m2*1e4:.2f} cm² ({config.S_skin_m2:.6f} m²)",
        "",
        "Skin Arc Lengths (root):",
        f"  L_(Skin root LE-FS)      = {config.L_Skin_LE_FS_mm:.2f} mm",
        f"  L_(Skin root LE-RS)      = {config.L_Skin_LE_RS_mm:.2f} mm",
        f"  L_(Skin root FS-RS)      = {config.L_Skin_FS_RS_mm:.2f} mm",
        "",
        "Rib Areas (root):",
        f"  S_(Rib LE-FS)            = {config.S_Rib_LE_FS_mm2:.2f} mm²",
        f"  S_(Rib FS-RS)            = {config.S_Rib_FS_RS_mm2:.2f} mm²",
        f"  S_Rib                    = {config.S_Rib_mm2:.2f} mm²",
    ]
    return "\n".join(lines)


def generate_mass_table(mass: MassBreakdown) -> str:
    """Generate mass breakdown table."""
    mass_g = mass.to_dict_grams()
    lines = [
        "",
        "MASS BREAKDOWN (Half-Wing)",
        "-" * 40,
        f"  {'Component':<20} {'Mass [g]':>12} {'Mass [kg]':>12}",
        f"  {'-'*20} {'-'*12} {'-'*12}",
    ]
    for comp, g in mass_g.items():
        kg = g / 1000
        lines.append(f"  {comp:<20} {g:>12.2f} {kg:>12.4f}")
    return "\n".join(lines)


def generate_stress_section(stress: StressResults) -> str:
    """Generate stress analysis section."""
    lines = [
        "",
        "STRUCTURAL PERFORMANCE",
        "-" * 40,
        "",
        "Skin Shear Stress:",
        f"  τ_skin_max               = {stress.tau_skin_max_MPa:.4f} MPa",
        f"  τ_allow                  = {stress.tau_skin_allow_MPa:.2f} MPa",
        f"  Safety                   = {stress.tau_skin_margin_percent:.1f}% ({stress.tau_skin_allow_MPa/stress.tau_skin_max_MPa:.2f}x)" if stress.tau_skin_max_MPa > 0 else f"  Safety                   = N/A",
        "",
        "Front Spar:",
        f"  σ_FS                     = {stress.sigma_b_FS_max_MPa:.2f} MPa",
        f"  τ_FS                     = {stress.tau_FS_max_MPa:.4f} MPa",
        f"  σ_vm                     = {stress.sigma_vm_FS_max_MPa:.2f} MPa",
        f"  σ_allow                  = {stress.sigma_FS_allow_MPa:.2f} MPa",
        f"  Safety                   = {stress.sigma_FS_margin_percent:.1f}% ({stress.sigma_FS_allow_MPa/stress.sigma_vm_FS_max_MPa:.2f}x)" if stress.sigma_vm_FS_max_MPa > 0 else f"  Safety                   = N/A",
        "",
        "Rear Spar:",
        f"  σ_RS                     = {stress.sigma_b_RS_max_MPa:.2f} MPa",
        f"  τ_RS                     = {stress.tau_RS_max_MPa:.4f} MPa",
        f"  σ_vm                     = {stress.sigma_vm_RS_max_MPa:.2f} MPa",
        f"  σ_allow                  = {stress.sigma_RS_allow_MPa:.2f} MPa",
        f"  Safety                   = {stress.sigma_RS_margin_percent:.1f}% ({stress.sigma_RS_allow_MPa/stress.sigma_vm_RS_max_MPa:.2f}x)" if stress.sigma_vm_RS_max_MPa > 0 else f"  Safety                   = N/A",
        "",
        "Tip Twist:",
        f"  θ_tip                    = {stress.theta_tip_deg:.3f}°",
        f"  θ_max                    = {stress.theta_max_deg:.1f}°",
        f"  Safety                   = {(1 - stress.theta_tip_deg/stress.theta_max_deg)*100:.1f}% ({stress.theta_max_deg/stress.theta_tip_deg:.2f}x)" if stress.theta_tip_deg > 0 else f"  Safety                   = N/A",
        "",
        f"Critical Station:          y = {stress.y_crit_mm:.1f} mm",
        f"Critical Component:        {stress.critical_component}",
    ]
    return "\n".join(lines)


def generate_root_reactions_section(reactions: RootReactionsReport) -> str:
    """Generate root reactions section."""
    lines = [
        "",
        "ROOT REACTION VECTORS",
        "-" * 40,
        "Force Vector [N]:",
        f"  [Fx, Fy, Fz] = [{reactions.Fx_N:.2f}, {reactions.Fy_N:.2f}, {reactions.Fz_N:.2f}]",
        "",
        "Moment Vector [N·m]:",
        f"  [Mx, My, Mz] = [{reactions.Mx_Nm:.3f}, {reactions.My_Nm:.4f}, {reactions.Mz_Nm:.3f}]",
        "",
        "Coordinate System (body axes):",
        "  x: aft (positive rearward)",
        "  y: starboard (positive right wing)",
        "  z: up (positive upward)",
    ]
    return "\n".join(lines)


def generate_materials_section(materials: Dict[str, str]) -> str:
    """Generate materials section."""
    lines = [
        "",
        "MATERIALS",
        "-" * 40,
    ]
    for component, material in materials.items():
        lines.append(f"  {component:<20} : {material}")
    return "\n".join(lines)


def generate_optimization_history_section(history: OptimizationHistory) -> str:
    """Generate optimization history section."""
    lines = [
        "",
        "OPTIMIZATION HISTORY",
        "-" * 40,
        f"  Total combinations       = {history.total_combinations}",
        f"  Valid geometry           = {history.valid_combinations}",
        f"  Accepted (passed all)    = {history.accepted_combinations}",
        "",
        "Rejection Reasons:",
    ]
    for reason, count in history.rejection_reasons.items():
        lines.append(f"  {reason:<30} : {count}")

    if history.best_solutions:
        lines.append("")
        lines.append("Top 10 Solutions (by mass):")
        lines.append(f"  {'Rank':<5} {'Mass[g]':>9} {'N_Rib':>6} {'X_FS[%]':>8} {'X_RS[%]':>8} {'d_FS[mm]':>9} {'t_FS[mm]':>9} {'d_RS[mm]':>9} {'t_RS[mm]':>9}")
        for i, sol in enumerate(history.best_solutions[:10], 1):
            lines.append(f"  {i:<5} {sol.get('mass_g', 0):>9.2f} {sol.get('N_Rib', 0):>6} "
                        f"{sol.get('X_FS', 0):>8.1f} {sol.get('X_RS', 0):>8.1f} "
                        f"{sol.get('d_FS_mm', 0):>9.2f} {sol.get('t_FS_mm', 0):>9.2f} "
                        f"{sol.get('d_RS_mm', 0):>9.2f} {sol.get('t_RS_mm', 0):>9.2f}")

    return "\n".join(lines)


def generate_full_report(config: OptimalConfig,
                         mass: MassBreakdown,
                         stress: StressResults,
                         reactions: RootReactionsReport,
                         materials: Dict[str, str],
                         history: Optional[OptimizationHistory] = None,
                         include_mass: bool = True,
                         include_history: bool = True,
                         phase_label: str = "") -> str:
    """Generate complete text report.

    Args:
        include_mass: If False, MASS BREAKDOWN section is omitted (caller prints it separately).
        include_history: If False, OPTIMIZATION HISTORY section is omitted (caller prints it separately).
        phase_label: Optional label like "PHASE-1" appended to the report header.
    """
    sections = [
        generate_header(phase_label),
        generate_optimal_params_section(config),
    ]

    if include_mass:
        sections.append(generate_mass_table(mass))

    sections.extend([
        generate_stress_section(stress),
        generate_root_reactions_section(reactions),
        generate_materials_section(materials),
    ])

    if include_history and history:
        sections.append(generate_optimization_history_section(history))

    if include_mass and include_history:
        sections.append("\n" + "=" * 70)
        sections.append("END OF REPORT")
        sections.append("=" * 70)

    return "\n".join(sections)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_json(data: Dict, filename: str):
    """Export data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Exported to: {filename}")


def export_to_csv(data: List[Dict], filename: str):
    """Export list of dicts to CSV file."""
    if not data:
        return

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"Exported to: {filename}")


def export_station_data(y: list, data_dict: Dict[str, list], filename: str):
    """Export station-by-station data to CSV."""
    rows = []
    for i in range(len(y)):
        row = {'y_mm': y[i] * 1000}
        for key, values in data_dict.items():
            row[key] = values[i]
        rows.append(row)

    export_to_csv(rows, filename)


# =============================================================================
# WARNING MESSAGES
# =============================================================================

def check_for_warnings(stress: StressResults, mass: MassBreakdown) -> List[str]:
    """Check for potential issues and generate warnings."""
    warnings = []

    # Check for very high safety margins (possible unit error)
    if stress.tau_skin_margin_percent > 99:
        warnings.append("WARNING: Skin shear stress margin > 99%. Check units!")

    if stress.sigma_FS_margin_percent > 99:
        warnings.append("WARNING: Front spar stress margin > 99%. Check units!")

    if stress.sigma_RS_margin_percent > 99:
        warnings.append("WARNING: Rear spar stress margin > 99%. Check units!")

    # Check for very low margins
    if stress.tau_skin_margin_percent < 10:
        warnings.append("WARNING: Skin shear stress margin < 10%. Design is marginal!")

    if stress.sigma_FS_margin_percent < 10:
        warnings.append("WARNING: Front spar stress margin < 10%. Design is marginal!")

    # Check for unrealistic mass
    if mass.m_total < 0.001:  # Less than 1 gram
        warnings.append("WARNING: Total mass < 1g. Check inputs!")

    if mass.m_total > 100:  # More than 100 kg for half-wing
        warnings.append("WARNING: Total mass > 100kg. Check inputs!")

    return warnings


if __name__ == "__main__":
    print("=== Reporting Module Test ===\n")

    # Create test data
    config = OptimalConfig(
        N_Rib=5,
        t_rib_mm=2.0,
        rib_spacing_mm=200.0,
        X_FS_percent=25.0,
        X_RS_percent=65.0,
        d_FS_outer_mm=20.0,
        t_FS_mm=2.0,
        d_RS_outer_mm=16.0,
        t_RS_mm=1.5,
        Lambda_FS_deg=2.15,
        Lambda_RS_deg=5.57,
        eta_FS_percent=71.9,
        eta_RS_percent=28.1,
        X_FS_mm=75.0,
        X_RS_mm=195.0,
        L_FS_mm=1000.0,
        L_RS_mm=1000.0,
        A_Act_FS_mm2=113.1,
        A_Act_RS_mm2=68.3,
        A_Cri_FS_mm2=50.0,
        A_Cri_RS_mm2=30.0,
        I_FS_mm4=4637.0,
        I_RS_mm4=1815.0,
        L_Skin_LE_FS_mm=75.0,
        L_Skin_LE_RS_mm=195.0,
        L_Skin_FS_RS_mm=120.0,
        S_Rib_LE_FS_mm2=2700.0,
        S_Rib_FS_RS_mm2=4320.0,
        S_Rib_mm2=7020.0
    )

    mass = MassBreakdown(
        m_skin=0.450,
        m_FS=0.318,
        m_RS=0.192,
        m_ribs=0.045,
        m_total=1.005
    )

    stress = StressResults(
        tau_skin_max_MPa=0.41,
        tau_skin_allow_MPa=220.7,
        tau_skin_margin_percent=99.8,
        sigma_vm_FS_max_MPa=31.0,
        sigma_FS_allow_MPa=381.3,
        sigma_FS_margin_percent=91.9,
        sigma_vm_RS_max_MPa=24.8,
        sigma_RS_allow_MPa=381.3,
        sigma_RS_margin_percent=93.5,
        y_crit_mm=0.0,
        critical_component="Front Spar"
    )

    reactions = RootReactionsReport(
        Fx_N=0.0,
        Fy_N=0.0,
        Fz_N=49.45,
        Mx_Nm=20.787,
        My_Nm=-3.545,
        Mz_Nm=0.0
    )

    materials = {
        "Spar": "AL7075-T6",
        "Skin": "AL7075-T6",
        "Rib": "PLA"
    }

    history = OptimizationHistory(
        total_combinations=1000,
        valid_combinations=800,
        accepted_combinations=150,
        rejection_reasons={
            "Skin stress exceeded": 400,
            "Spar stress exceeded": 200,
            "Deflection exceeded": 50
        },
        best_solutions=[
            {"mass_g": 1005.0, "N_Rib": 5, "d_FS_mm": 20.0, "d_RS_mm": 16.0},
            {"mass_g": 1010.0, "N_Rib": 5, "d_FS_mm": 20.0, "d_RS_mm": 18.0},
            {"mass_g": 1020.0, "N_Rib": 6, "d_FS_mm": 20.0, "d_RS_mm": 16.0},
        ]
    )

    # Generate report
    report = generate_full_report(config, mass, stress, reactions, materials, history)
    print(report)

    # Check warnings
    warnings = check_for_warnings(stress, mass)
    if warnings:
        print("\n" + "!" * 70)
        for w in warnings:
            print(w)
        print("!" * 70)

    # Test export
    print("\n--- Testing Export ---")
    export_to_json({"config": asdict(config), "mass": asdict(mass)}, "test_results.json")

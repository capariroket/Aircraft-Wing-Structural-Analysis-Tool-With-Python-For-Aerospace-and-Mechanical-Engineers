"""
torsion.py - Wing-Box Torsion and Shear Flow Module

Implements Bredt-Batho theory for closed-section torsion.
Supports single-cell and multi-cell analysis.

All internal calculations use SI units (N, m, Pa).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ShearFlowResult:
    """Shear flow result at a station."""
    y: float            # Spanwise position [m]
    T: float            # Applied torsion [N·m]
    A_m: float          # Enclosed area [m²]
    q: float            # Shear flow [N/m]
    tau_skin: float     # Skin shear stress [Pa]
    twist_rate: float   # Twist rate dθ/dz [rad/m] (optional)


# =============================================================================
# SINGLE-CELL BREDT-BATHO
# =============================================================================

def bredt_batho_single_cell(T: float, A_m: float) -> float:
    """
    Single-cell Bredt-Batho shear flow.

    q = T / (2 * A_m)

    Args:
        T: Applied torsion [N·m]
        A_m: Enclosed (median) area [m²]

    Returns:
        Shear flow q [N/m]
    """
    if A_m <= 0:
        raise ValueError(f"Enclosed area must be positive, got {A_m}")
    return T / (2 * A_m)


def shear_stress_from_flow(q: float, t: float) -> float:
    """
    Convert shear flow to shear stress.

    τ = q / t

    Args:
        q: Shear flow [N/m]
        t: Wall thickness [m]

    Returns:
        Shear stress [Pa]
    """
    if t <= 0:
        raise ValueError(f"Thickness must be positive, got {t}")
    return q / t


def twist_rate_single_cell(q: float, A_m: float, G: float,
                            perimeter: float, t_avg: float) -> float:
    """
    Compute twist rate for single cell.

    dθ/dz = q * ∮(ds/t) / (2 * A_m * G)

    For uniform thickness:
    dθ/dz = q * P / (2 * A_m * G * t)

    Args:
        q: Shear flow [N/m]
        A_m: Enclosed area [m²]
        G: Shear modulus [Pa]
        perimeter: Cell perimeter [m]
        t_avg: Average wall thickness [m]

    Returns:
        Twist rate [rad/m]
    """
    delta = perimeter / t_avg  # ∮ ds/t
    return (q * delta) / (2 * A_m * G)


def analyze_single_cell(T: float, A_m: float, t_skin: float,
                        G: float, perimeter: float,
                        y: float = 0.0) -> ShearFlowResult:
    """
    Complete single-cell torsion analysis.

    Args:
        T: Applied torsion [N·m]
        A_m: Enclosed area [m²]
        t_skin: Skin thickness [m]
        G: Shear modulus [Pa]
        perimeter: Cell perimeter [m]
        y: Spanwise position [m]

    Returns:
        ShearFlowResult with all computed values
    """
    q = bredt_batho_single_cell(T, A_m)
    tau = shear_stress_from_flow(q, t_skin)
    twist = twist_rate_single_cell(q, A_m, G, perimeter, t_skin)

    return ShearFlowResult(
        y=y,
        T=T,
        A_m=A_m,
        q=q,
        tau_skin=tau,
        twist_rate=twist
    )


# =============================================================================
# MULTI-CELL BREDT-BATHO
# =============================================================================

@dataclass
class CellGeometry:
    """Geometry of a single cell in multi-cell section."""
    A: float                # Enclosed area [m²]
    walls: List[Tuple[float, float, float]]  # [(length, thickness, G), ...]
    # Each wall: (ds, t, G) - segment length, thickness, shear modulus


def compute_delta(walls: List[Tuple[float, float, float]], G_ref: float) -> float:
    """
    Compute ∮ ds/(t*) for a cell, where t* = (G/G_ref) * t.

    Args:
        walls: List of (length, thickness, G) for each wall segment
        G_ref: Reference shear modulus [Pa]

    Returns:
        δ = ∮ ds/t* [1/m]
    """
    delta = 0.0
    for ds, t, G in walls:
        t_star = (G / G_ref) * t
        delta += ds / t_star
    return delta


def solve_multicell_torsion(T: float, cells: List[CellGeometry],
                             shared_walls: List[Tuple[int, int, float, float, float]],
                             G_ref: float) -> Tuple[np.ndarray, float]:
    """
    Solve multi-cell torsion problem.

    System of equations:
    1. Torque equilibrium: T = Σ (2 * A_R * q_R)
    2. Compatibility: All cells have same twist rate

    Args:
        T: Total applied torsion [N·m]
        cells: List of CellGeometry for each cell
        shared_walls: List of (cell_i, cell_j, ds, t, G) for shared walls
        G_ref: Reference shear modulus [Pa]

    Returns:
        Tuple of (q_array, twist_rate)
        q_array: Shear flows for each cell [N/m]
        twist_rate: Common twist rate [rad/m]
    """
    n = len(cells)

    if n == 1:
        # Single cell case
        q = T / (2 * cells[0].A)
        delta = compute_delta(cells[0].walls, G_ref)
        twist = (q * delta) / (2 * cells[0].A * G_ref)
        return np.array([q]), twist

    # Build coefficient matrix for compatibility equations
    # Row i: twist rate equation for cell i
    # Extra row: torque equilibrium
    # Columns: q_0, q_1, ..., q_{n-1}, twist_rate

    A_mat = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    # Compute delta for each cell (own walls only)
    deltas = [compute_delta(cell.walls, G_ref) for cell in cells]

    # Build shared wall lookup
    # shared_wall_delta[i][j] = delta contribution from wall between cell i and j
    shared_delta = {}
    for ci, cj, ds, t, G in shared_walls:
        t_star = (G / G_ref) * t
        delta_ij = ds / t_star
        shared_delta[(ci, cj)] = delta_ij
        shared_delta[(cj, ci)] = delta_ij

    # Compatibility equations (rows 0 to n-1)
    # dθ/dz = (1/(2*A_R*G_ref)) * (q_R * δ_R - Σ q_neighbor * δ_shared)
    # All equal, so: cell 0 twist = cell i twist for i > 0

    for i in range(n):
        # Coefficient for own cell
        A_mat[i, i] = deltas[i] / (2 * cells[i].A)

        # Coefficients for neighbors (shared walls)
        for j in range(n):
            if i != j and (i, j) in shared_delta:
                # Shared wall contributes negatively
                A_mat[i, j] -= shared_delta[(i, j)] / (2 * cells[i].A)

        # Coefficient for twist rate (on RHS, moved to LHS as negative)
        A_mat[i, n] = -G_ref

    # Torque equilibrium (last row)
    # T = Σ 2 * A_R * q_R
    for i in range(n):
        A_mat[n, i] = 2 * cells[i].A

    b_vec[n] = T

    # Solve system
    try:
        x = np.linalg.solve(A_mat, b_vec)
        q_array = x[:n]
        twist_rate = x[n]
        return q_array, twist_rate
    except np.linalg.LinAlgError:
        raise ValueError("Could not solve multi-cell system (singular matrix)")


# =============================================================================
# PRACTICAL WING-BOX ANALYSIS (2-CELL)
# =============================================================================

@dataclass
class TwoCellBoxGeometry:
    """Two-cell wing box geometry (FS-RS with middle spar or front/rear cells)."""
    A_front: float      # Front cell enclosed area [m²]
    A_rear: float       # Rear cell enclosed area [m²]
    # Front cell walls: top skin, bottom skin, front spar
    L_front_top: float  # Front cell top skin length [m]
    L_front_bot: float  # Front cell bottom skin length [m]
    L_FS: float         # Front spar height [m]
    # Rear cell walls: top skin, bottom skin, rear spar
    L_rear_top: float   # Rear cell top skin length [m]
    L_rear_bot: float   # Rear cell bottom skin length [m]
    L_RS: float         # Rear spar height [m]
    # Middle spar (shared wall)
    L_MS: float         # Middle spar height [m]
    # Thicknesses
    t_skin: float       # Skin thickness [m]
    t_spar: float       # Spar web thickness [m]
    # Material
    G: float            # Shear modulus [Pa]


def analyze_two_cell_box(T: float, geom: TwoCellBoxGeometry) -> Tuple[float, float, float]:
    """
    Analyze two-cell wing box torsion.

    Args:
        T: Applied torsion [N·m]
        geom: Two-cell geometry

    Returns:
        Tuple of (q_front, q_rear, twist_rate)
    """
    # Build cells
    front_walls = [
        (geom.L_front_top, geom.t_skin, geom.G),
        (geom.L_front_bot, geom.t_skin, geom.G),
        (geom.L_FS, geom.t_spar, geom.G),
    ]
    rear_walls = [
        (geom.L_rear_top, geom.t_skin, geom.G),
        (geom.L_rear_bot, geom.t_skin, geom.G),
        (geom.L_RS, geom.t_spar, geom.G),
    ]

    cells = [
        CellGeometry(A=geom.A_front, walls=front_walls),
        CellGeometry(A=geom.A_rear, walls=rear_walls),
    ]

    # Shared wall (middle spar)
    shared_walls = [
        (0, 1, geom.L_MS, geom.t_spar, geom.G)
    ]

    q_array, twist_rate = solve_multicell_torsion(T, cells, shared_walls, geom.G)

    return q_array[0], q_array[1], twist_rate


# =============================================================================
# STATION-BY-STATION ANALYSIS
# =============================================================================

def analyze_all_stations(y: np.ndarray, T: np.ndarray, A_m: np.ndarray,
                         t_skin: float, G: float,
                         box_width: np.ndarray, box_height: np.ndarray
                         ) -> List[ShearFlowResult]:
    """
    Analyze shear flow at all rib stations (single-cell).

    Args:
        y: Spanwise positions [m]
        T: Torsion at each station [N·m]
        A_m: Enclosed area at each station [m²]
        t_skin: Skin thickness [m]
        G: Shear modulus [Pa]
        box_width: Wing-box width at each station [m]
        box_height: Wing-box height at each station [m]

    Returns:
        List of ShearFlowResult for each station
    """
    results = []

    for i in range(len(y)):
        # Approximate perimeter (rectangular)
        perimeter = 2 * (box_width[i] + box_height[i])

        result = analyze_single_cell(
            T=T[i],
            A_m=A_m[i],
            t_skin=t_skin,
            G=G,
            perimeter=perimeter,
            y=y[i]
        )
        results.append(result)

    return results


def find_critical_station(results: List[ShearFlowResult]) -> Tuple[int, ShearFlowResult]:
    """
    Find the station with maximum shear stress.

    Args:
        results: List of ShearFlowResult

    Returns:
        Tuple of (index, critical_result)
    """
    max_idx = 0
    max_tau = abs(results[0].tau_skin)

    for i, r in enumerate(results):
        if abs(r.tau_skin) > max_tau:
            max_tau = abs(r.tau_skin)
            max_idx = i

    return max_idx, results[max_idx]


if __name__ == "__main__":
    from geometry import PlanformParams, SparPosition, compute_all_stations
    from loads import (FlightCondition, AeroCenter, analyze_loads,
                       LoadDistributionType, PitchMomentDistributionType)
    from materials import MaterialDatabase
    import numpy as np

    print("=== Torsion Module Test ===\n")

    # Setup
    planform = PlanformParams.from_input(
        b=2.0, AR=8.0, taper_ratio=0.5, t_c=0.12,
        S_ref=0.5, C_r_mm=300, c_MGC=0.25, Y_bar_mm=400
    )

    flight = FlightCondition(
        W0=50, n=2.0, V_c=25, rho=1.225,
        C_m=-0.05, S_ref=0.5, c_MGC=0.25
    )

    spar_pos = SparPosition(X_FS_percent=25, X_RS_percent=65)
    ac = AeroCenter(x_ac_percent=25, Lambda_ac_deg=0)

    # Material
    db = MaterialDatabase()
    skin_mat = db.get_material('AL7075-T6')
    G = skin_mat.G
    t_skin = 0.001  # 1 mm

    # Geometry at stations
    N_Rib = 10
    stations = compute_all_stations(planform, spar_pos, N_Rib)

    y = np.array([s.y for s in stations])
    chord = np.array([s.chord for s in stations])
    x_FS = np.array([s.x_FS for s in stations])
    x_RS = np.array([s.x_RS for s in stations])
    A_m = np.array([s.A_m for s in stations])
    box_width = np.array([s.width for s in stations])
    box_height = np.array([s.h_box for s in stations])

    # Load analysis
    loads = analyze_loads(y, chord, x_FS, x_RS, flight, ac, planform.L_span,
                          LoadDistributionType.ELLIPTIC,
                          PitchMomentDistributionType.CHORD_WEIGHTED)

    # Torsion analysis at all stations
    print("Single-cell torsion analysis:")
    results = analyze_all_stations(y, loads.T, A_m, t_skin, G, box_width, box_height)

    print(f"\n{'y[mm]':>8} {'T[N·m]':>10} {'q[N/m]':>10} {'τ[MPa]':>10} {'dθ/dz[°/m]':>12}")
    for r in results:
        tau_mpa = r.tau_skin / 1e6
        twist_deg = np.degrees(r.twist_rate)
        print(f"{r.y*1000:8.1f} {r.T:10.4f} {r.q:10.2f} {tau_mpa:10.4f} {twist_deg:12.4f}")

    # Critical station
    crit_idx, crit = find_critical_station(results)
    print(f"\nCritical station: y = {crit.y*1000:.1f} mm")
    print(f"  τ_max = {crit.tau_skin/1e6:.4f} MPa")
    print(f"  q_max = {crit.q:.2f} N/m")

    # Check against allowable
    SF = 1.5
    tau_allow = skin_mat.tau_u / SF
    margin = (tau_allow - abs(crit.tau_skin)) / tau_allow * 100
    print(f"\nAllowable τ (SF={SF}): {tau_allow/1e6:.1f} MPa")
    print(f"Safety margin: {margin:.1f}%")

    # Simple single-cell test
    print("\n--- Simple Single-Cell Test ---")
    T_test = 100  # N·m
    A_test = 0.01  # m²
    q_test = bredt_batho_single_cell(T_test, A_test)
    print(f"T = {T_test} N·m, A_m = {A_test} m²")
    print(f"q = T/(2*A_m) = {q_test} N/m")
    print(f"Expected: {T_test/(2*A_test)} N/m")

"""
loads.py - Load Distribution and Internal Forces Module

Handles lift distribution, shear force, bending moment,
pitching moment, and torsion calculations along the half-span.

All internal calculations use SI units (N, m, N·m).
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from enum import Enum


class LoadDistributionType(Enum):
    """Load distribution type selection."""
    UNIFORM = "uniform"
    ELLIPTIC = "elliptic"
    SIMPLIFIED = "simplified"


class PitchMomentDistributionType(Enum):
    """Pitching moment distribution type."""
    UNIFORM = "uniform"
    CHORD_WEIGHTED = "chord_weighted"


@dataclass
class FlightCondition:
    """Flight condition parameters."""
    W0: float       # MTOW [N]
    n: float        # Load factor [-]
    V_c: float      # Flight velocity [m/s]
    rho: float      # Air density [kg/m³]
    C_m: float      # Pitching moment coefficient [-]
    S_ref: float    # Reference wing area [m²]
    c_MGC: float    # Mean geometric chord [m]

    @property
    def L_total(self) -> float:
        """Total lift required [N]."""
        return self.n * self.W0

    @property
    def L_half(self) -> float:
        """Half-wing lift [N]."""
        return self.L_total / 2

    @property
    def q_inf(self) -> float:
        """Dynamic pressure [Pa]."""
        return 0.5 * self.rho * self.V_c**2

    @property
    def M_pitch_total(self) -> float:
        """Total pitching moment [N·m]."""
        return self.C_m * self.q_inf * self.S_ref * self.c_MGC

    @property
    def M_pitch_half(self) -> float:
        """Half-wing pitching moment [N·m]."""
        return self.M_pitch_total / 2


# =============================================================================
# LIFT DISTRIBUTION FUNCTIONS
# =============================================================================

def lift_distribution_uniform(y: np.ndarray, L_half: float, L_span: float) -> np.ndarray:
    """
    Uniform lift distribution.

    w(y) = L_half / L_span

    Args:
        y: Spanwise positions [m]
        L_half: Half-wing total lift [N]
        L_span: Half-span [m]

    Returns:
        Distributed load w(y) [N/m]
    """
    return np.full_like(y, L_half / L_span)


def lift_distribution_elliptic(y: np.ndarray, L_half: float, L_span: float) -> np.ndarray:
    """
    Elliptic lift distribution.

    w(y) = w0 * sqrt(1 - (y/L_span)²)
    w0 = 4 * L_half / (π * L_span)

    Args:
        y: Spanwise positions [m]
        L_half: Half-wing total lift [N]
        L_span: Half-span [m]

    Returns:
        Distributed load w(y) [N/m]
    """
    w0 = 4 * L_half / (np.pi * L_span)
    # Proper handling: clip eta to [0,1] and use maximum to avoid sqrt of negative
    eta = np.clip(y / L_span, 0.0, 1.0)
    w = w0 * np.sqrt(np.maximum(0.0, 1 - eta**2))
    # Explicitly set tip value to zero for exact elliptic distribution
    if isinstance(y, np.ndarray) and len(y) > 0:
        w[-1] = 0.0
    return w


def get_lift_distribution_func(dist_type: LoadDistributionType) -> Callable:
    """Get the appropriate lift distribution function."""
    if dist_type == LoadDistributionType.UNIFORM:
        return lift_distribution_uniform
    elif dist_type == LoadDistributionType.ELLIPTIC:
        return lift_distribution_elliptic
    elif dist_type == LoadDistributionType.SIMPLIFIED:
        # Simplified uses elliptic shape for station-by-station values
        # but root reactions are overridden by the closed-form formulas.
        return lift_distribution_elliptic
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


# =============================================================================
# SHEAR FORCE AND BENDING MOMENT (Numerical Integration)
# =============================================================================


def root_bending_moment_from_total_load(n: float, W0: float, y_bar_m: float) -> float:
    """
    Root bending moment (single value) using:
        M_root = n * W0 * y_bar / 2

    Parameters
    ----------
    n : float
        Load factor [-]
    W0 : float
        Total weight or total lift being reacted by the wing [N]
        (use consistent convention with your report)
    y_bar_m : float
        Mean spanwise moment arm [m]

    Returns
    -------
    float
        Root bending moment [N*m]
    """
    return n * W0 * (y_bar_m / 2.0)


def root_bending_moment_ybar_in_mm(n: float, W0: float, y_bar_mm: float) -> float:
    """
    Same as above, but y_bar is given in mm (as in your screenshot sometimes).
    Converts mm -> m internally.
    """
    y_bar_m = y_bar_mm / 1000.0
    return n * W0 * (y_bar_m / 2.0)









def compute_shear_force(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute shear force distribution V(y).

    V(y) = ∫_y^{L_span} w(ξ) dξ

    Integrates from tip to root (cantilever beam).

    Args:
        y: Spanwise positions [m] (must be sorted ascending)
        w: Distributed load [N/m]

    Returns:
        Shear force V(y) [N]
    """
    # Integrate from tip to each station
    # V(y) = integral from y to L_span of w
    V = np.zeros_like(y)

    # Use trapezoidal integration from tip (last element) to each station
    for i in range(len(y) - 2, -1, -1):
        # Integrate from y[i] to y[i+1]
        dy = y[i + 1] - y[i]
        V[i] = V[i + 1] + 0.5 * (w[i] + w[i + 1]) * dy

    return V


def compute_bending_moment(y: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute bending moment distribution M(y).

    M(y) = ∫_y^{L_span} V(ξ) dξ

    Args:
        y: Spanwise positions [m]
        V: Shear force [N]

    Returns:
        Bending moment M(y) [N·m]
    """
    M = np.zeros_like(y)

    for i in range(len(y) - 2, -1, -1):
        dy = y[i + 1] - y[i]
        M[i] = M[i + 1] + 0.5 * (V[i] + V[i + 1]) * dy

    return M


# =============================================================================
# PITCHING MOMENT DISTRIBUTION
# =============================================================================

def pitch_moment_distribution_uniform(y: np.ndarray, M_pitch_half: float,
                                       L_span: float, chord_func: Callable = None) -> np.ndarray:
    """
    Uniform pitching moment distribution.

    m_pitch(y) = M_pitch_half / L_span [N·m per m]

    Args:
        y: Spanwise positions [m]
        M_pitch_half: Half-wing pitching moment [N·m]
        L_span: Half-span [m]
        chord_func: Not used for uniform (kept for interface consistency)

    Returns:
        Distributed pitching moment [N·m/m]
    """
    return np.full_like(y, M_pitch_half / L_span)


def pitch_moment_distribution_chord_weighted(y: np.ndarray, M_pitch_half: float,
                                              L_span: float, chord_func: Callable) -> np.ndarray:
    """
    Chord-weighted pitching moment distribution.

    m_pitch(y) = k * c(y)
    where k is determined by ∫ m_pitch(y) dy = M_pitch_half

    Args:
        y: Spanwise positions [m]
        M_pitch_half: Half-wing pitching moment [N·m]
        L_span: Half-span [m]
        chord_func: Function that returns chord at y

    Returns:
        Distributed pitching moment [N·m/m]
    """
    chord = chord_func(y)

    # Compute integral of chord using trapezoidal rule
    chord_integral = np.trapz(chord, y)

    if chord_integral == 0:
        return np.zeros_like(y)

    k = M_pitch_half / chord_integral
    return k * chord


# =============================================================================
# TORSION CALCULATIONS
# =============================================================================

@dataclass
class AeroCenter:
    """Aerodynamic center parameters."""
    x_ac_percent: float  # AC position [% chord]
    Lambda_ac_deg: float  # Sweep angle at AC [deg]

    def x_ac_at_station(self, chord: float) -> float:
        """AC x-position at given chord [m]."""
        return (self.x_ac_percent / 100) * chord


def compute_shear_center(x_FS: np.ndarray, x_RS: np.ndarray) -> np.ndarray:
    """
    Compute shear center position (preliminary: midpoint of spars).

    x_sc(y) ≈ (x_FS(y) + x_RS(y)) / 2

    Args:
        x_FS: Front spar x-positions [m]
        x_RS: Rear spar x-positions [m]

    Returns:
        Shear center x-positions [m]
    """
    return (x_FS + x_RS) / 2


def compute_eccentricity(x_ac: np.ndarray, x_sc: np.ndarray) -> np.ndarray:
    """
    Compute eccentricity between AC and shear center.

    e(y) = x_ac(y) - x_sc(y)

    Positive e means AC is ahead of SC (nose-up torsion from lift).

    Args:
        x_ac: Aerodynamic center x-positions [m]
        x_sc: Shear center x-positions [m]

    Returns:
        Eccentricity [m]
    """
    return x_ac - x_sc


def compute_torsion_intensity(w: np.ndarray, e: np.ndarray,
                               m_pitch: np.ndarray) -> np.ndarray:
    """
    Compute torsion intensity (distributed torque).

    t(y) = t_L(y) + t_M(y)
    t_L(y) = w(y) * e(y)  (lift-induced)
    t_M(y) = m_pitch(y)   (pitching moment)

    Sign convention:
    - Positive t = nose-up torque

    Args:
        w: Lift distribution [N/m]
        e: Eccentricity [m]
        m_pitch: Pitching moment distribution [N·m/m]

    Returns:
        Torsion intensity [N·m/m]
    """
    t_L = w * e
    return t_L + m_pitch


def compute_torsion(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute torsion distribution T(y).

    T(y) = ∫_y^{L_span} t(ξ) dξ

    Args:
        y: Spanwise positions [m]
        t: Torsion intensity [N·m/m]

    Returns:
        Torsion T(y) [N·m]
    """
    T = np.zeros_like(y)

    for i in range(len(y) - 2, -1, -1):
        dy = y[i + 1] - y[i]
        T[i] = T[i + 1] + 0.5 * (t[i] + t[i + 1]) * dy

    return T


# =============================================================================
# COMPLETE LOAD ANALYSIS
# =============================================================================

@dataclass
class LoadResults:
    """Complete load analysis results."""
    y: np.ndarray           # Spanwise positions [m]
    w: np.ndarray           # Lift distribution [N/m]
    V: np.ndarray           # Shear force [N]
    M: np.ndarray           # Bending moment [N·m]
    m_pitch: np.ndarray     # Pitching moment distribution [N·m/m]
    t: np.ndarray           # Torsion intensity [N·m/m]
    T: np.ndarray           # Torsion [N·m]
    e: np.ndarray           # Eccentricity [m]
    x_ac: np.ndarray        # Aero center positions [m]
    x_sc: np.ndarray        # Shear center positions [m]
    # Simplified-mode override values (None = use numerical integration results)
    M_root_simplified: float = None  # Closed-form M_root override [N·m]
    T_root_simplified: float = None  # Closed-form T_root override [N·m]

    @property
    def V_root(self) -> float:
        """Root shear force [N]."""
        return self.V[0]

    @property
    def M_root(self) -> float:
        """Root bending moment [N·m].
        Returns closed-form value if simplified mode was used."""
        if self.M_root_simplified is not None:
            return self.M_root_simplified
        return self.M[0]

    @property
    def T_root(self) -> float:
        """Root torsion [N·m].
        Returns closed-form value if simplified mode was used."""
        if self.T_root_simplified is not None:
            return self.T_root_simplified
        return self.T[0]


def analyze_loads(y: np.ndarray,
                  chord: np.ndarray,
                  x_FS: np.ndarray,
                  x_RS: np.ndarray,
                  flight: FlightCondition,
                  ac: AeroCenter,
                  L_span: float,
                  load_dist_type: LoadDistributionType = LoadDistributionType.ELLIPTIC,
                  pitch_dist_type: PitchMomentDistributionType = PitchMomentDistributionType.CHORD_WEIGHTED,
                  Y_bar_m: float = None,
                  ) -> LoadResults:
    """
    Perform complete load analysis.

    Args:
        y: Spanwise positions [m]
        chord: Chord at each station [m]
        x_FS: Front spar positions [m]
        x_RS: Rear spar positions [m]
        flight: Flight condition parameters
        ac: Aerodynamic center parameters
        L_span: Half-span [m]
        load_dist_type: Lift distribution type
        pitch_dist_type: Pitching moment distribution type
        Y_bar_m: Mean aerodynamic centre spanwise position [m].
                 Required when load_dist_type == SIMPLIFIED.

    Returns:
        LoadResults with all computed values
    """
    # Lift distribution
    lift_func = get_lift_distribution_func(load_dist_type)
    w = lift_func(y, flight.L_half, L_span)

    # Shear and bending
    V = compute_shear_force(y, w)
    M = compute_bending_moment(y, V)

    # Pitching moment distribution
    def chord_func(y_val):
        return np.interp(y_val, y, chord)

    if pitch_dist_type == PitchMomentDistributionType.UNIFORM:
        m_pitch = pitch_moment_distribution_uniform(y, flight.M_pitch_half, L_span)
    else:
        m_pitch = pitch_moment_distribution_chord_weighted(y, flight.M_pitch_half,
                                                           L_span, chord_func)

    # Torsion
    x_ac = np.array([ac.x_ac_at_station(c) for c in chord])
    x_sc = compute_shear_center(x_FS, x_RS)
    e = compute_eccentricity(x_ac, x_sc)
    t = compute_torsion_intensity(w, e, m_pitch)
    T = compute_torsion(y, t)

    # -----------------------------------------------------------------
    # SIMPLIFIED mode: override M_root and T_root with closed-form
    # formulas from the design specification:
    #
    #   M_root = n · W₀ · Ȳ / 2
    #
    #   T_root = 1000 · 0.5 · ρ · V_c² · (S_ref / 2) · (Ȳ / 1000) · C_m
    #          = 0.5 · ρ · V_c² · (S_ref / 2) · Ȳ · C_m
    #          (the 1000 and /1000 cancel when Ȳ is in metres)
    #
    # The full M and T *arrays* are also scaled so that M[0] == M_root_simplified
    # and T[0] == T_root_simplified, preserving the shape of the distribution.
    # This ensures spar/torsion station-by-station analysis is consistent.
    # -----------------------------------------------------------------
    M_root_simplified = None
    T_root_simplified = None
    if load_dist_type == LoadDistributionType.SIMPLIFIED:
        if Y_bar_m is None:
            raise ValueError(
                "Y_bar_m (mean aerodynamic centre spanwise position) "
                "must be provided for SIMPLIFIED load distribution."
            )
        M_root_simplified = flight.n * flight.W0 * Y_bar_m / 2.0
        q_inf = 0.5 * flight.rho * flight.V_c ** 2
        T_root_simplified = q_inf * (flight.S_ref / 2.0) * Y_bar_m * flight.C_m

        # Scale the M array so that M[0] equals M_root_simplified
        if M[0] != 0.0:
            M = M * (M_root_simplified / M[0])
        else:
            M[0] = M_root_simplified

        # Scale the T array so that T[0] equals T_root_simplified
        if T[0] != 0.0:
            T = T * (T_root_simplified / T[0])
        else:
            T[0] = T_root_simplified

    return LoadResults(
        y=y, w=w, V=V, M=M,
        m_pitch=m_pitch, t=t, T=T,
        e=e, x_ac=x_ac, x_sc=x_sc,
        M_root_simplified=M_root_simplified,
        T_root_simplified=T_root_simplified,
    )


# =============================================================================
# ROOT FORCE/MOMENT VECTORS
# =============================================================================

@dataclass
class RootReactions:
    """Root reaction forces and moments in body axes."""
    Fx: float  # Chordwise shear (from drag, typically small) [N]
    Fy: float  # Spanwise shear (typically 0 for symmetric flight) [N]
    Fz: float  # Vertical shear (= V_root for level flight) [N]
    Mx: float  # Roll moment (= M_root for level flight) [N·m]
    My: float  # Pitch moment (= T_root) [N·m]
    Mz: float  # Yaw moment (typically 0) [N·m]


def compute_root_reactions(loads: LoadResults, drag_estimate: float = 0.0) -> RootReactions:
    """
    Compute root reaction vectors.

    Coordinate system (body axes):
    - x: aft (positive rearward)
    - y: starboard (positive right wing)
    - z: up (positive upward)

    Args:
        loads: LoadResults from analysis
        drag_estimate: Estimated drag force [N] (optional)

    Returns:
        RootReactions with force/moment vectors
    """
    return RootReactions(
        Fx=drag_estimate,     # Chordwise (drag)
        Fy=0.0,               # Spanwise (symmetric)
        Fz=loads.V_root,      # Vertical (shear)
        Mx=loads.M_root,      # Roll (bending)
        My=loads.T_root,      # Pitch (torsion)
        Mz=0.0                # Yaw (symmetric)
    )


if __name__ == "__main__":
    from geometry import PlanformParams, SparPosition, generate_rib_stations, chord_at_station

    print("=== Loads Module Test ===\n")

    # Setup planform
    planform = PlanformParams.from_input(
        b=4.093, AR=11.03, taper_ratio=0.649, t_c=0.17,
        S_ref=1.519, C_r_mm=450.12, c_MGC=376.7, Y_bar_mm=950.65
    )

    # Flight condition (small UAV, 2g maneuver)
    flight = FlightCondition(
        W0=290.260,          # 50 N (~5 kg)
        n=2.0,          # 2g
        V_c=21.48,         # 25 m/s
        rho=1.112,      # sea level
        C_m=-0.262,      # typical
        S_ref=1.519,
        c_MGC=276.7
    )

    print(f"Flight condition:")
    print(f"  W0 = {flight.W0} N")
    print(f"  n = {flight.n}")
    print(f"  L_half = {flight.L_half} N")
    print(f"  q_inf = {flight.q_inf:.1f} Pa")
    print(f"  M_pitch_half = {flight.M_pitch_half:.3f} N·m")

    # Spar positions
    spar_pos = SparPosition(X_FS_percent=15, X_RS_percent=60)

    # Generate stations
    N_Rib = 12
    y = generate_rib_stations(N_Rib, planform.L_span)
    chord = np.array([chord_at_station(yi, planform.C_r, planform.c_tip, planform.L_span) for yi in y])
    x_FS = np.array([spar_pos.x_FS_at_station(c) for c in chord])
    x_RS = np.array([spar_pos.x_RS_at_station(c) for c in chord])

    # Aerodynamic center
    ac = AeroCenter(x_ac_percent=25, Lambda_ac_deg=0)

    # Analyze with elliptic distribution
    print("\n--- Elliptic Load Distribution ---")
    loads = analyze_loads(y, chord, x_FS, x_RS, flight, ac, planform.L_span,
                          LoadDistributionType.ELLIPTIC,
                          PitchMomentDistributionType.CHORD_WEIGHTED)

    print(f"\nLoad distribution at stations:")
    print(f"  {'y[mm]':>8} {'w[N/m]':>10} {'V[N]':>10} {'M[N·m]':>10} {'T[N·m]':>10}")
    for i in range(0, len(y), 2):
        print(f"  {y[i]*1000:8.1f} {loads.w[i]:10.2f} {loads.V[i]:10.2f} {loads.M[i]:10.3f} {loads.T[i]:10.4f}")

    print(f"\nRoot reactions:")
    print(f"  V_root = {loads.V_root:.2f} N")
    print(f"  M_root = {loads.M_root:.3f} N·m")
    print(f"  T_root = {loads.T_root:.4f} N·m")

    # Root reaction vectors
    reactions = compute_root_reactions(loads)
    print(f"\nRoot reaction vectors:")
    print(f"  F = [{reactions.Fx:.2f}, {reactions.Fy:.2f}, {reactions.Fz:.2f}] N")
    print(f"  M = [{reactions.Mx:.3f}, {reactions.My:.4f}, {reactions.Mz:.3f}] N·m")

    # Compare uniform vs elliptic
    print("\n--- Comparison: Uniform vs Elliptic ---")
    loads_uniform = analyze_loads(y, chord, x_FS, x_RS, flight, ac, planform.L_span,
                                   LoadDistributionType.UNIFORM,
                                   PitchMomentDistributionType.UNIFORM)

    print(f"  {'Distribution':<15} {'V_root[N]':>12} {'M_root[N·m]':>12} {'T_root[N·m]':>12}")
    print(f"  {'Uniform':<15} {loads_uniform.V_root:12.2f} {loads_uniform.M_root:12.3f} {loads_uniform.T_root:12.4f}")
    print(f"  {'Elliptic':<15} {loads.V_root:12.2f} {loads.M_root:12.3f} {loads.T_root:12.4f}")

"""
ribs.py - Rib Geometry and Mass Module

Handles rib spacing, area calculations, and mass estimation.

All internal calculations use SI units (m, m², kg).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional

from buckling import (skin_shear_buckling_critical, skin_compression_buckling_critical,
                      combined_interaction_ratio, rib_web_shear_buckling_critical,
                      compute_s_max_shear, compute_s_max_compression, check_bay_buckling)


@dataclass
class RibProperties:
    """Rib material and geometry properties."""
    t_rib: float        # Rib thickness [m]
    density: float      # Rib material density [kg/m³]
    sigma_u: float      # Ultimate tensile strength [Pa]
    tau_u: float        # Ultimate shear strength [Pa]

    @classmethod
    def from_mm(cls, t_rib_mm: float, material) -> 'RibProperties':
        """
        Create from millimeter input and material object.

        Args:
            t_rib_mm: Rib thickness [mm]
            material: Material object with density, sigma_u, tau_u

        Returns:
            RibProperties instance
        """
        return cls(
            t_rib=t_rib_mm / 1000,
            density=material.density,
            sigma_u=material.sigma_u,
            tau_u=material.tau_u
        )


@dataclass
class RibGeometry:
    """Geometry of a single rib at a station."""
    y: float            # Spanwise position [m]
    chord: float        # Local chord [m]
    h_box: float        # Wing-box height [m]
    x_FS: float         # Front spar position [m]
    x_RS: float         # Rear spar position [m]
    parabolic: bool = False  # Use parabolic profile (True) or rectangular (False)

    @property
    def S_LE_FS(self) -> float:
        """Area of rib section from LE to front spar [m²]."""
        if self.parabolic:
            return (2 / 3) * self.x_FS * self.h_box
        return self.h_box * self.x_FS

    @property
    def S_FS_RS(self) -> float:
        """Area of rib section between spars (wing-box) [m²]."""
        if self.parabolic:
            S_LE_RS = (2 / 3) * self.x_RS * self.h_box
            return S_LE_RS - self.S_LE_FS
        return self.h_box * (self.x_RS - self.x_FS)

    @property
    def S_total(self) -> float:
        """Total rib area (simplified, up to rear spar) [m²]."""
        return self.S_LE_FS + self.S_FS_RS

    def mass(self, t_rib: float, density: float) -> float:
        """
        Calculate rib mass.

        m = S * t * ρ

        Args:
            t_rib: Rib thickness [m]
            density: Material density [kg/m³]

        Returns:
            Rib mass [kg]
        """
        return self.S_total * t_rib * density


def compute_rib_spacing(L_span: float, N_Rib: int) -> float:
    """
    Compute rib spacing.

    spacing = L_span / N_Rib

    Args:
        L_span: Half-span [m]
        N_Rib: Number of rib bays

    Returns:
        Rib spacing [m]
    """
    if N_Rib < 1:
        raise ValueError("N_Rib must be at least 1")
    return L_span / N_Rib


def generate_rib_geometries(y_stations: np.ndarray,
                            chord: np.ndarray,
                            h_box: np.ndarray,
                            x_FS: np.ndarray,
                            x_RS: np.ndarray,
                            parabolic: bool = False) -> List[RibGeometry]:
    """
    Generate RibGeometry for each station.

    Args:
        y_stations: Spanwise positions [m]
        chord: Chord at each station [m]
        h_box: Wing-box height at each station [m]
        x_FS: Front spar position at each station [m]
        x_RS: Rear spar position at each station [m]
        parabolic: Use parabolic profile (True) or rectangular (False)

    Returns:
        List of RibGeometry objects
    """
    ribs = []
    for i in range(len(y_stations)):
        rib = RibGeometry(
            y=y_stations[i],
            chord=chord[i],
            h_box=h_box[i],
            x_FS=x_FS[i],
            x_RS=x_RS[i],
            parabolic=parabolic
        )
        ribs.append(rib)
    return ribs


@dataclass
class RibMassResult:
    """Result of rib mass calculation."""
    N_Rib: int              # Number of rib bays
    spacing: float          # Rib spacing [m]
    total_rib_mass: float   # Total mass of all ribs [kg]
    avg_rib_mass: float     # Average mass per rib [kg]
    root_rib_area: float    # Root rib area [m²]
    tip_rib_area: float     # Tip rib area [m²]


def compute_total_rib_mass(ribs: List[RibGeometry],
                           rib_props: RibProperties,
                           L_span: float) -> RibMassResult:
    """
    Compute total mass of all ribs.

    Note: N_Rib bays means N_Rib+1 rib stations, but typically
    we count N_Rib actual ribs (excluding root which is part of fuselage).

    For this preliminary analysis, we assume N_Rib ribs (not N_Rib+1).

    Args:
        ribs: List of RibGeometry for each station
        rib_props: Rib material/thickness properties
        L_span: Half-span [m]

    Returns:
        RibMassResult with mass breakdown
    """
    N_Rib = len(ribs) - 1  # Number of bays (stations - 1)

    if N_Rib < 1:
        raise ValueError("Need at least 2 stations for rib calculation")

    # Calculate mass for each rib (exclude root rib, index 0)
    # In practice, ribs are at stations 1 to N_Rib
    total_mass = 0.0
    for i in range(1, len(ribs)):  # Skip root
        rib_mass = ribs[i].mass(rib_props.t_rib, rib_props.density)
        total_mass += rib_mass

    spacing = compute_rib_spacing(L_span, N_Rib)
    avg_mass = total_mass / N_Rib if N_Rib > 0 else 0

    return RibMassResult(
        N_Rib=N_Rib,
        spacing=spacing,
        total_rib_mass=total_mass,
        avg_rib_mass=avg_mass,
        root_rib_area=ribs[0].S_total,
        tip_rib_area=ribs[-1].S_total
    )


def compute_rib_mass_simple(N_Rib: int, S_rib_avg: float,
                            t_rib: float, density: float) -> float:
    """
    Simple rib mass calculation (uniform ribs).

    m_ribs = N_Rib * S_rib * t_rib * ρ

    Args:
        N_Rib: Number of ribs
        S_rib_avg: Average rib area [m²]
        t_rib: Rib thickness [m]
        density: Material density [kg/m³]

    Returns:
        Total rib mass [kg]
    """
    return N_Rib * S_rib_avg * t_rib * density


# =============================================================================
# RIB STRESS (SIMPLIFIED)
# =============================================================================

def rib_shear_stress_estimate(V_local: float, S_rib: float, t_rib: float) -> float:
    """
    Estimate average shear stress in rib (very simplified).

    τ_rib ≈ V / (S_rib * some_factor)

    This is a rough estimate. For preliminary sizing, we use:
    τ_rib ≈ V / (h_box * t_rib)

    Args:
        V_local: Local shear force carried by rib [N]
        S_rib: Rib area [m²]
        t_rib: Rib thickness [m]

    Returns:
        Estimated shear stress [Pa]
    """
    # Simplified: assume shear is carried by rib web area
    # Web area ≈ h_box * t_rib, but we don't have h_box here
    # Use S_rib as proxy (conservative)
    if S_rib <= 0 or t_rib <= 0:
        return 0.0
    return V_local / (S_rib * 0.5)  # Factor 0.5 is empirical


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

@dataclass
class RibOutputs:
    """Rib-related outputs for reporting."""
    N_Rib: int
    spacing_mm: float
    t_rib_mm: float
    S_Rib_LE_FS_mm2: float      # Root rib area LE-FS [mm²]
    S_Rib_FS_RS_mm2: float      # Root rib area FS-RS [mm²]
    S_Rib_total_mm2: float      # Root rib total area [mm²]
    total_mass_kg: float
    total_mass_g: float


def format_rib_outputs(ribs: List[RibGeometry],
                       rib_props: RibProperties,
                       L_span: float) -> RibOutputs:
    """
    Format rib outputs for reporting.

    Args:
        ribs: List of RibGeometry
        rib_props: Rib properties
        L_span: Half-span [m]

    Returns:
        RibOutputs with formatted values
    """
    mass_result = compute_total_rib_mass(ribs, rib_props, L_span)
    root_rib = ribs[0]

    return RibOutputs(
        N_Rib=mass_result.N_Rib,
        spacing_mm=mass_result.spacing * 1000,
        t_rib_mm=rib_props.t_rib * 1000,
        S_Rib_LE_FS_mm2=root_rib.S_LE_FS * 1e6,
        S_Rib_FS_RS_mm2=root_rib.S_FS_RS * 1e6,
        S_Rib_total_mm2=root_rib.S_total * 1e6,
        total_mass_kg=mass_result.total_rib_mass,
        total_mass_g=mass_result.total_rib_mass * 1000
    )


# =============================================================================
# ADAPTIVE RIB INSERTION ALGORITHM
# =============================================================================

@dataclass
class AdaptiveRibConfig:
    """Configuration for adaptive rib insertion.

    Contains material and geometry properties needed for buckling checks,
    plus algorithm control parameters.
    """
    E_skin: float           # Skin Young's modulus [Pa]
    nu_skin: float          # Skin Poisson's ratio [-]
    t_skin: float           # Skin thickness [m]
    E_rib: float            # Rib Young's modulus [Pa]
    nu_rib: float           # Rib Poisson's ratio [-]
    t_rib: float            # Rib thickness [m]
    tau_allow_skin: float   # Allowable skin shear stress [Pa]
    sigma_allow_skin: float # Allowable skin compression stress [Pa]
    k_s: float = 10       # Shear buckling coefficient (SS long plate)
    k_c: float = 8        # Compression buckling coefficient (SS uniaxial)
    s_min: float = 0.020    # Minimum practical rib spacing [m] (20 mm)
    margin: float = 0.95    # Safety margin factor on s_target
    buckling_mode: int = 1  # 1=shear only, 2=shear+compression
    N_Rib_max: int = 0      # Maximum rib count (0=unlimited)


@dataclass
class BayResult:
    """Result for a single panel bay in the bay results table.

    Stores geometry, loads, stresses, buckling criticals, margins, and
    pass/fail flags for every bay between consecutive ribs.
    """
    bay_id: int
    y_start: float          # [m]
    y_end: float            # [m]
    spacing: float          # [m]
    y_mid: float            # [m]
    chord_mid: float        # [m]
    h_box_mid: float        # [m]
    A_m_mid: float          # [m^2] enclosed area at midpoint
    V_mid: float            # [N]
    M_mid: float            # [N.m]
    T_mid: float            # [N.m]
    tau_skin: float         # [Pa]
    sigma_comp: float       # [Pa] skin compression from bending
    tau_allow: float        # [Pa]
    sigma_allow: float      # [Pa]
    tau_cr: float           # [Pa] shear buckling critical
    sigma_cr: float         # [Pa] compression buckling critical
    R_combined: float       # [-] interaction ratio
    skin_margin_allow: float      # [-] tau_allow / tau_panel
    skin_margin_buckling: float   # [-] tau_cr / tau_panel
    comp_margin_allow: float      # [-] sigma_allow / sigma_comp
    comp_margin_buckling: float   # [-] sigma_cr / sigma_comp
    sigma_vm_FS: float      # [Pa]
    sigma_vm_RS: float      # [Pa]
    pass_shear_buckling: bool
    pass_comp_buckling: bool
    pass_combined: bool
    pass_shear_allow: bool
    pass_comp_allow: bool
    pass_all: bool


def _check_bay_buckling(config: AdaptiveRibConfig, spacing: float,
                        tau_panel: float, sigma_panel: float) -> Tuple[bool, float]:
    """Check if a bay passes buckling for given spacing.

    Returns (passes, R_combined) where R_combined is the interaction ratio.
    In shear-only mode (buckling_mode=1), compression buckling is skipped.
    """
    tau_cr = skin_shear_buckling_critical(
        config.E_skin, config.nu_skin, config.t_skin, spacing, config.k_s)

    fail_shear = (tau_panel > tau_cr)

    if config.buckling_mode == 2:
        # Shear + compression
        sigma_cr = skin_compression_buckling_critical(
            config.E_skin, config.nu_skin, config.t_skin, spacing, config.k_c)
        fail_comp = (sigma_panel > sigma_cr)
        R = combined_interaction_ratio(tau_panel, tau_cr, sigma_panel, sigma_cr)
        fail_combined = (R >= 1.0)
        passes = not (fail_shear or fail_comp or fail_combined)
    else:
        # Shear only
        sigma_cr = float('inf')
        R = (tau_panel / tau_cr) ** 2 if tau_cr > 0 else 0.0
        passes = not fail_shear

    return passes, R


def _find_max_feasible_spacing(config: AdaptiveRibConfig,
                                tau_panel: float, sigma_panel: float,
                                s_initial: float, tol: float = 0.02) -> float:
    """Binary search for maximum feasible spacing in a bay (constant stress).

    Returns max feasible spacing [m], or 0 if infeasible even at s_min.
    """
    passes, _ = _check_bay_buckling(config, s_initial, tau_panel, sigma_panel)
    if passes:
        return s_initial

    s_lower = 0.0
    s_upper = s_initial
    s_try = s_initial * 0.5

    MAX_BISECT = 50
    for _ in range(MAX_BISECT):
        if s_try < config.s_min:
            passes_at_smin, _ = _check_bay_buckling(config, config.s_min,
                                                     tau_panel, sigma_panel)
            if passes_at_smin:
                s_lower = config.s_min
            break
        passes, _ = _check_bay_buckling(config, s_try, tau_panel, sigma_panel)
        if passes:
            s_lower = s_try
            break
        else:
            s_upper = s_try
            s_try = s_try * 0.5

    if s_lower == 0:
        return 0.0

    for _ in range(MAX_BISECT):
        if s_upper <= 0 or (s_upper - s_lower) / s_upper < tol:
            break
        s_mid = (s_lower + s_upper) / 2.0
        passes, _ = _check_bay_buckling(config, s_mid, tau_panel, sigma_panel)
        if passes:
            s_lower = s_mid
        else:
            s_upper = s_mid

    return s_lower


def _find_max_feasible_spacing_at_y(config: AdaptiveRibConfig,
                                     y_left: float, s_max: float,
                                     tau_skin_at_y: Callable[[float], float],
                                     sigma_comp_at_y: Callable[[float], float],
                                     tol: float = 0.02) -> float:
    """Binary search for max feasible spacing using real stress at each candidate position.

    Unlike _find_max_feasible_spacing which uses constant stress, this version
    evaluates the actual stress at y_left and y_left+s for each candidate spacing.
    This gives correct results when stress varies along the span.

    Args:
        config: AdaptiveRibConfig.
        y_left: Left boundary of the bay [m].
        s_max: Maximum spacing to try (distance to right boundary) [m].
        tau_skin_at_y: Interpolation function for tau_skin.
        sigma_comp_at_y: Interpolation function for sigma_comp.
        tol: Convergence tolerance.

    Returns:
        Max feasible spacing [m], or 0 if infeasible even at s_min.
    """
    def _check_at_spacing(s):
        y_right = y_left + s
        tau_p = max(abs(tau_skin_at_y(y_left)), abs(tau_skin_at_y(y_right)))
        sig_p = max(abs(sigma_comp_at_y(y_left)), abs(sigma_comp_at_y(y_right)))
        return _check_bay_buckling(config, s, tau_p, sig_p)

    # Check if current spacing already passes
    passes, _ = _check_at_spacing(s_max)
    if passes:
        return s_max

    # Phase 1: Halve until we find a passing spacing
    s_lower = 0.0
    s_upper = s_max
    s_try = s_max * 0.5

    MAX_BISECT = 50
    for _ in range(MAX_BISECT):
        if s_try < config.s_min:
            passes_at_smin, _ = _check_at_spacing(config.s_min)
            if passes_at_smin:
                s_lower = config.s_min
            break
        passes, _ = _check_at_spacing(s_try)
        if passes:
            s_lower = s_try
            break
        else:
            s_upper = s_try
            s_try = s_try * 0.5

    if s_lower == 0:
        return 0.0

    # Phase 2: Binary search between s_lower (PASS) and s_upper (FAIL)
    for _ in range(MAX_BISECT):
        if s_upper <= 0 or (s_upper - s_lower) / s_upper < tol:
            break
        s_mid = (s_lower + s_upper) / 2.0
        passes, _ = _check_at_spacing(s_mid)
        if passes:
            s_lower = s_mid
        else:
            s_upper = s_mid

    return s_lower


def _compute_bay_margin(config: AdaptiveRibConfig, spacing: float,
                         tau_panel: float, sigma_panel: float) -> float:
    """Compute buckling margin for a bay. Lower = more critical."""
    tau_cr = skin_shear_buckling_critical(
        config.E_skin, config.nu_skin, config.t_skin, spacing, config.k_s)
    margin_shear = tau_cr / tau_panel if tau_panel > 0 else float('inf')

    if config.buckling_mode == 2:
        sigma_cr = skin_compression_buckling_critical(
            config.E_skin, config.nu_skin, config.t_skin, spacing, config.k_c)
        margin_comp = sigma_cr / sigma_panel if sigma_panel > 0 else float('inf')
        return min(margin_shear, margin_comp)
    else:
        return margin_shear


def adaptive_rib_insertion(y_ribs: np.ndarray, config: AdaptiveRibConfig,
                           tau_skin_at_y: Callable[[float], float],
                           sigma_comp_at_y: Callable[[float], float],
                           V_at_y: Callable[[float], float],
                           h_box_at_y: Callable[[float], float]
                           ) -> Tuple[np.ndarray, bool, str]:
    """
    Sweep-based adaptive rib placement from root to tip.

    Places ribs iteratively: starting at y=0, finds the maximum feasible
    spacing (limited only by buckling) at the current position, places a
    new rib there, and repeats until the wingtip is reached.

    No artificial s_max cap — each bay is as wide as buckling allows.
    Stress is evaluated at both bay boundaries (conservative envelope).

    Algorithm:
        1. Start at y_left = 0.
        2. Check if the remaining span [y_left, y_tip] passes as a single bay.
           If yes, done.
        3. Binary-search for the largest spacing s such that the bay
           [y_left, y_left+s] passes all buckling checks.
        4. Place a new rib at y_left + s.
        5. Set y_left = y_left + s and go to step 2.

    Args:
        y_ribs: Initial rib positions [m] (only y_ribs[0] and y_ribs[-1]
                are used as root and tip boundaries).
        config: AdaptiveRibConfig with material/geometry and control parameters.
        tau_skin_at_y: Callable(y) -> tau_skin [Pa].
        sigma_comp_at_y: Callable(y) -> sigma_comp [Pa].
        V_at_y: Callable(y) -> V [N]. Shear force interpolation (unused,
                kept for API compatibility).
        h_box_at_y: Callable(y) -> h_box [m]. Box height interpolation
                (unused, kept for API compatibility).

    Returns:
        Tuple of (final_y_ribs, feasible, message).
    """
    MAX_BISECT = 50
    y_sorted = np.sort(np.array(y_ribs, dtype=float))
    y_tip = y_sorted[-1]

    ribs = [0.0]

    while ribs[-1] < y_tip - config.s_min:
        y_left = ribs[-1]
        remaining = y_tip - y_left

        # Check if remaining distance passes as a single bay
        tau_p = max(abs(tau_skin_at_y(y_left)), abs(tau_skin_at_y(y_tip)))
        sig_p = max(abs(sigma_comp_at_y(y_left)), abs(sigma_comp_at_y(y_tip)))
        passes, _ = _check_bay_buckling(config, remaining, tau_p, sig_p)
        if passes:
            break  # Rest of span is one bay — done

        # Binary search for max feasible spacing
        def _check_spacing(s):
            yr = y_left + s
            tp = max(abs(tau_skin_at_y(y_left)), abs(tau_skin_at_y(yr)))
            sp = max(abs(sigma_comp_at_y(y_left)), abs(sigma_comp_at_y(yr)))
            return _check_bay_buckling(config, s, tp, sp)

        # Phase 1: halve until we find a passing spacing
        s_lower = 0.0
        s_upper = remaining
        s_try = remaining * 0.5

        for _ in range(MAX_BISECT):
            if s_try < config.s_min:
                p, _ = _check_spacing(config.s_min)
                if p:
                    s_lower = config.s_min
                break
            p, _ = _check_spacing(s_try)
            if p:
                s_lower = s_try
                break
            else:
                s_upper = s_try
                s_try *= 0.5

        if s_lower == 0:
            # Infeasible even at s_min
            msg = (f"INFEASIBLE at y={y_left*1000:.1f}mm. "
                   f"{len(ribs)} ribs placed so far.")
            ribs.append(y_tip)
            return np.array(ribs), False, msg

        # Phase 2: binary search refine between s_lower (PASS) and s_upper (FAIL)
        for _ in range(MAX_BISECT):
            if s_upper <= 0 or (s_upper - s_lower) / s_upper < 0.02:
                break
            s_mid = (s_lower + s_upper) / 2.0
            p, _ = _check_spacing(s_mid)
            if p:
                s_lower = s_mid
            else:
                s_upper = s_mid

        new_rib = y_left + s_lower

        # If very close to tip, let last bay reach tip
        if (y_tip - new_rib) < config.s_min:
            break

        ribs.append(new_rib)

        # N_Rib_max check (number of bays = len(ribs))
        if config.N_Rib_max > 0 and len(ribs) >= config.N_Rib_max:
            break

    # Close with tip
    ribs.append(y_tip)
    y_arr = np.array(ribs)
    n_bays = len(y_arr) - 1

    # Final verification — check every bay
    all_pass = True
    fail_bays = []
    for i in range(n_bays):
        ys, ye = y_arr[i], y_arr[i + 1]
        sp = ye - ys
        tp = max(abs(tau_skin_at_y(ys)), abs(tau_skin_at_y(ye)))
        sgp = max(abs(sigma_comp_at_y(ys)), abs(sigma_comp_at_y(ye)))
        p, _ = _check_bay_buckling(config, sp, tp, sgp)
        if not p:
            all_pass = False
            fail_bays.append(i + 1)

    if all_pass:
        msg = (f"FEASIBLE: {len(y_arr)} rib stations ({n_bays} bays). "
               f"Sweep placement complete.")
    else:
        msg = (f"INFEASIBLE: {len(y_arr)} rib stations ({n_bays} bays). "
               f"Failing bays: {fail_bays}.")

    return y_arr, all_pass, msg


def compute_bay_results(y_ribs: np.ndarray,
                        config: AdaptiveRibConfig,
                        tau_skin_at_y: Callable[[float], float],
                        sigma_comp_at_y: Callable[[float], float],
                        V_at_y: Callable[[float], float],
                        M_at_y: Callable[[float], float],
                        T_at_y: Callable[[float], float],
                        chord_at_y: Callable[[float], float],
                        h_box_at_y: Callable[[float], float],
                        A_m_at_y: Callable[[float], float],
                        sigma_vm_FS_at_y: Optional[Callable[[float], float]] = None,
                        sigma_vm_RS_at_y: Optional[Callable[[float], float]] = None
                        ) -> List[BayResult]:
    """
    Compute detailed bay-by-bay results for the final rib layout.

    For each bay (panel between consecutive ribs), evaluates midpoint conditions
    including geometry, loads, stresses, buckling criticals, margins, and
    pass/fail flags.

    Args:
        y_ribs: Final rib positions [m] (sorted).
        config: AdaptiveRibConfig with material/geometry parameters.
        tau_skin_at_y: Callable(y) -> tau_skin [Pa].
        sigma_comp_at_y: Callable(y) -> sigma_comp [Pa].
        V_at_y: Callable(y) -> V [N] shear force.
        M_at_y: Callable(y) -> M [N.m] bending moment.
        T_at_y: Callable(y) -> T [N.m] torsion.
        chord_at_y: Callable(y) -> chord [m].
        h_box_at_y: Callable(y) -> h_box [m].
        A_m_at_y: Callable(y) -> A_m [m^2] enclosed area.
        sigma_vm_FS_at_y: Optional callable(y) -> sigma_vm [Pa] for front spar.
        sigma_vm_RS_at_y: Optional callable(y) -> sigma_vm [Pa] for rear spar.

    Returns:
        List of BayResult for each bay.
    """
    results = []
    y = np.sort(y_ribs)

    for i in range(len(y) - 1):
        y_start = y[i]
        y_end = y[i + 1]
        spacing = y_end - y_start
        y_mid = (y_start + y_end) / 2.0

        # Geometry at midpoint
        chord_mid = chord_at_y(y_mid)
        h_box_mid = h_box_at_y(y_mid)
        A_m_mid = A_m_at_y(y_mid)

        # Loads at midpoint
        V_mid = V_at_y(y_mid)
        M_mid = M_at_y(y_mid)
        T_mid = T_at_y(y_mid)

        # Skin stresses: use maximum of endpoints for buckling (conservative)
        tau_panel = max(abs(tau_skin_at_y(y_start)),
                        abs(tau_skin_at_y(y_end)))
        sigma_panel = max(abs(sigma_comp_at_y(y_start)),
                          abs(sigma_comp_at_y(y_end)))

        # Buckling criticals for this bay's spacing
        tau_cr = skin_shear_buckling_critical(
            config.E_skin, config.nu_skin, config.t_skin,
            spacing, config.k_s)
        sigma_cr = skin_compression_buckling_critical(
            config.E_skin, config.nu_skin, config.t_skin,
            spacing, config.k_c)

        # Combined interaction
        R = combined_interaction_ratio(tau_panel, tau_cr,
                                       sigma_panel, sigma_cr)

        # Margins
        skin_margin_allow = (config.tau_allow_skin / tau_panel
                             if tau_panel > 0 else float('inf'))
        skin_margin_buckling = (tau_cr / tau_panel
                                if tau_panel > 0 else float('inf'))
        comp_margin_allow = (config.sigma_allow_skin / sigma_panel
                             if sigma_panel > 0 else float('inf'))
        comp_margin_buckling = (sigma_cr / sigma_panel
                                if sigma_panel > 0 else float('inf'))

        # Spar von Mises at midpoint (optional)
        vm_fs = sigma_vm_FS_at_y(y_mid) if sigma_vm_FS_at_y else 0.0
        vm_rs = sigma_vm_RS_at_y(y_mid) if sigma_vm_RS_at_y else 0.0

        # Pass/fail flags
        pass_shear_buckling = (tau_panel <= tau_cr)
        pass_shear_allow = (tau_panel <= config.tau_allow_skin)
        pass_comp_allow = (sigma_panel <= config.sigma_allow_skin)

        if config.buckling_mode == 2:
            # Shear + compression mode
            pass_comp_buckling = (sigma_panel <= sigma_cr)
            pass_combined = (R < 1.0)
        else:
            # Shear only mode — compression buckling always passes
            pass_comp_buckling = True
            pass_combined = True

        pass_all_flag = (pass_shear_buckling and pass_comp_buckling and
                         pass_combined and pass_shear_allow and pass_comp_allow)

        results.append(BayResult(
            bay_id=i + 1,
            y_start=y_start,
            y_end=y_end,
            spacing=spacing,
            y_mid=y_mid,
            chord_mid=chord_mid,
            h_box_mid=h_box_mid,
            A_m_mid=A_m_mid,
            V_mid=V_mid,
            M_mid=M_mid,
            T_mid=T_mid,
            tau_skin=tau_panel,
            sigma_comp=sigma_panel,
            tau_allow=config.tau_allow_skin,
            sigma_allow=config.sigma_allow_skin,
            tau_cr=tau_cr,
            sigma_cr=sigma_cr,
            R_combined=R,
            skin_margin_allow=skin_margin_allow,
            skin_margin_buckling=skin_margin_buckling,
            comp_margin_allow=comp_margin_allow,
            comp_margin_buckling=comp_margin_buckling,
            sigma_vm_FS=vm_fs,
            sigma_vm_RS=vm_rs,
            pass_shear_buckling=pass_shear_buckling,
            pass_comp_buckling=pass_comp_buckling,
            pass_combined=pass_combined,
            pass_shear_allow=pass_shear_allow,
            pass_comp_allow=pass_comp_allow,
            pass_all=pass_all_flag,
        ))

    return results


def format_bay_results_csv(bay_results: List[BayResult]) -> str:
    """
    Format bay results as a CSV string.

    Each row represents one panel bay. Columns cover geometry, loads,
    stresses, buckling criticals, margins, and pass/fail flags.

    Args:
        bay_results: List of BayResult from compute_bay_results().

    Returns:
        CSV-formatted string with header row.
    """
    header = (
        "Bay,y_start[m],y_end[m],spacing[mm],y_mid[m],"
        "chord_mid[mm],h_box_mid[mm],A_m_mid[mm2],"
        "V_mid[N],M_mid[N.m],T_mid[N.m],"
        "tau_skin[MPa],sigma_comp[MPa],"
        "tau_allow[MPa],sigma_allow[MPa],"
        "tau_cr[MPa],sigma_cr[MPa],R_combined,"
        "MoS_shear_allow,MoS_shear_buckle,"
        "MoS_comp_allow,MoS_comp_buckle,"
        "sigma_vm_FS[MPa],sigma_vm_RS[MPa],"
        "pass_shear_buckle,pass_comp_buckle,pass_combined,"
        "pass_shear_allow,pass_comp_allow,pass_all"
    )
    lines = [header]

    for br in bay_results:
        row = (
            f"{br.bay_id},"
            f"{br.y_start:.6f},{br.y_end:.6f},{br.spacing*1000:.2f},{br.y_mid:.6f},"
            f"{br.chord_mid*1000:.2f},{br.h_box_mid*1000:.2f},{br.A_m_mid*1e6:.2f},"
            f"{br.V_mid:.2f},{br.M_mid:.4f},{br.T_mid:.4f},"
            f"{br.tau_skin/1e6:.4f},{br.sigma_comp/1e6:.4f},"
            f"{br.tau_allow/1e6:.4f},{br.sigma_allow/1e6:.4f},"
            f"{br.tau_cr/1e6:.4f},{br.sigma_cr/1e6:.4f},{br.R_combined:.6f},"
            f"{br.skin_margin_allow:.4f},{br.skin_margin_buckling:.4f},"
            f"{br.comp_margin_allow:.4f},{br.comp_margin_buckling:.4f},"
            f"{br.sigma_vm_FS/1e6:.4f},{br.sigma_vm_RS/1e6:.4f},"
            f"{br.pass_shear_buckling},{br.pass_comp_buckling},{br.pass_combined},"
            f"{br.pass_shear_allow},{br.pass_comp_allow},{br.pass_all}"
        )
        lines.append(row)

    return "\n".join(lines)


def format_bay_results_table(bay_results: List[BayResult]) -> str:
    """
    Format bay results as a human-readable table for terminal output.

    Shows key columns: bay ID, spacing, tau_skin, tau_cr, sigma_comp,
    sigma_cr, R_combined, and pass/fail status.

    Args:
        bay_results: List of BayResult from compute_bay_results().

    Returns:
        Formatted table string.
    """
    lines = []
    sep = "-" * 120

    lines.append(sep)
    lines.append(
        f"{'Bay':>4} {'y_start':>8} {'y_end':>8} {'s[mm]':>7} "
        f"{'tau[MPa]':>9} {'tau_cr':>9} {'sig[MPa]':>9} {'sig_cr':>9} "
        f"{'R_comb':>7} {'MoS_sh':>7} {'MoS_co':>7} {'Status':>8}"
    )
    lines.append(sep)

    n_pass = 0
    n_fail = 0

    for br in bay_results:
        status = "PASS" if br.pass_all else "FAIL"
        if br.pass_all:
            n_pass += 1
        else:
            n_fail += 1

        lines.append(
            f"{br.bay_id:4d} "
            f"{br.y_start*1000:8.1f} {br.y_end*1000:8.1f} "
            f"{br.spacing*1000:7.1f} "
            f"{br.tau_skin/1e6:9.3f} {br.tau_cr/1e6:9.3f} "
            f"{br.sigma_comp/1e6:9.3f} {br.sigma_cr/1e6:9.3f} "
            f"{br.R_combined:7.4f} "
            f"{br.skin_margin_buckling:7.2f} {br.comp_margin_buckling:7.2f} "
            f"{status:>8}"
        )

    lines.append(sep)
    lines.append(
        f"Summary: {n_pass} PASS, {n_fail} FAIL out of "
        f"{len(bay_results)} bays  |  "
        f"Total ribs: {len(bay_results) + 1}"
    )
    lines.append(sep)

    return "\n".join(lines)


@dataclass
class RibWebBucklingResult:
    """Result of rib web buckling check and optional t_rib thickening.

    For each rib station, stores the shear stress, critical stress,
    margin, and the final t_rib needed to pass.
    """
    y_positions: np.ndarray          # Rib spanwise positions [m]
    h_box_at_station: np.ndarray     # Box height at each station [m]
    t_rib_per_station: np.ndarray    # Final t_rib at each station [m]
    tau_rib: np.ndarray              # Rib web shear stress [Pa]
    tau_cr_rib: np.ndarray           # Rib web critical shear buckling [Pa]
    mos_rib: np.ndarray              # Margin of safety (tau_cr / tau_rib)
    pass_flags: np.ndarray           # Bool: True if passes
    t_rib_initial_mm: float          # Initial t_rib [mm]
    t_rib_max_mm: float              # Max t_rib after thickening [mm]
    n_thickened: int                 # Number of stations that needed thickening
    all_pass: bool                   # True if all stations pass


def fix_rib_web_buckling(y_ribs: np.ndarray,
                         V_at_y: Callable[[float], float],
                         h_box_at_y: Callable[[float], float],
                         E_rib: float, nu_rib: float,
                         t_rib_initial: float,
                         t_rib_step: float = 0.0005,
                         t_rib_max: float = 0.010,
                         k_s_rib: float = 5.34) -> RibWebBucklingResult:
    """
    Check rib web shear buckling at each rib station and thicken as needed.

    At each rib position, computes:
        tau_rib = V / (h_box * t_rib)
        tau_cr  = k_s * pi^2 * E / (12*(1-nu^2)) * (t_rib / h_box)^2

    If tau_rib > tau_cr, increases t_rib by t_rib_step (default 0.5mm)
    until it passes or t_rib_max is reached.

    Only failing stations get thicker ribs — passing ones keep the original.

    Args:
        y_ribs: Rib positions [m].
        V_at_y: Callable(y) -> V [N], shear force interpolation.
        h_box_at_y: Callable(y) -> h_box [m], box height interpolation.
        E_rib: Rib Young's modulus [Pa].
        nu_rib: Rib Poisson's ratio [-].
        t_rib_initial: Initial rib thickness [m].
        t_rib_step: Thickness increment [m] (default 0.5mm).
        t_rib_max: Maximum allowable rib thickness [m] (default 10mm).
        k_s_rib: Rib web shear buckling coefficient (default 5.34, SS).

    Returns:
        RibWebBucklingResult with per-station results.
    """
    n = len(y_ribs)
    t_rib_arr = np.full(n, t_rib_initial)
    h_box_arr = np.zeros(n)
    tau_rib_arr = np.zeros(n)
    tau_cr_arr = np.zeros(n)
    mos_arr = np.zeros(n)
    pass_arr = np.ones(n, dtype=bool)
    n_thickened = 0

    for i, y_pos in enumerate(y_ribs):
        h = float(h_box_at_y(y_pos))
        V = float(abs(V_at_y(y_pos)))
        h_box_arr[i] = h

        if h <= 0 or V <= 0:
            tau_rib_arr[i] = 0.0
            tau_cr_arr[i] = float('inf')
            mos_arr[i] = float('inf')
            continue

        t_rib = t_rib_initial
        while t_rib <= t_rib_max:
            tau_rib = V / (h * t_rib)
            tau_cr = k_s_rib * np.pi**2 * E_rib / (12 * (1 - nu_rib**2)) * (t_rib / h)**2

            if tau_rib <= tau_cr:
                break
            t_rib += t_rib_step

        tau_rib_final = V / (h * t_rib)
        tau_cr_final = k_s_rib * np.pi**2 * E_rib / (12 * (1 - nu_rib**2)) * (t_rib / h)**2

        t_rib_arr[i] = t_rib
        tau_rib_arr[i] = tau_rib_final
        tau_cr_arr[i] = tau_cr_final
        mos_arr[i] = tau_cr_final / tau_rib_final if tau_rib_final > 0 else float('inf')
        pass_arr[i] = tau_rib_final <= tau_cr_final

        if t_rib > t_rib_initial:
            n_thickened += 1

    return RibWebBucklingResult(
        y_positions=y_ribs,
        h_box_at_station=h_box_arr,
        t_rib_per_station=t_rib_arr,
        tau_rib=tau_rib_arr,
        tau_cr_rib=tau_cr_arr,
        mos_rib=mos_arr,
        pass_flags=pass_arr,
        t_rib_initial_mm=t_rib_initial * 1000,
        t_rib_max_mm=float(np.max(t_rib_arr)) * 1000,
        n_thickened=n_thickened,
        all_pass=bool(np.all(pass_arr)),
    )


def format_rib_web_buckling_table(rwb: RibWebBucklingResult,
                                   rib_geometries: Optional[List] = None,
                                   rib_density: float = 0.0) -> str:
    """Format rib web buckling results as a human-readable table.

    Args:
        rwb: RibWebBucklingResult from fix_rib_web_buckling().
        rib_geometries: Optional list of RibGeometry objects for mass calculation.
        rib_density: Rib material density [kg/m3] for mass calculation.

    Returns:
        Formatted table string.
    """
    show_mass = rib_geometries is not None and rib_density > 0
    lines = []

    if show_mass:
        sep = "-" * 116
        lines.append(sep)
        lines.append(
            f"{'Rib':>4} {'y[mm]':>8} {'h_box[mm]':>10} {'t_rib[mm]':>10} "
            f"{'tau_rib[MPa]':>13} {'tau_cr[MPa]':>12} {'MoS':>8} {'Mass[g]':>9} {'Status':>8}"
        )
    else:
        sep = "-" * 100
        lines.append(sep)
        lines.append(
            f"{'Rib':>4} {'y[mm]':>8} {'h_box[mm]':>10} {'t_rib[mm]':>10} "
            f"{'tau_rib[MPa]':>13} {'tau_cr[MPa]':>12} {'MoS':>8} {'Status':>8}"
        )
    lines.append(sep)

    n_pass = 0
    n_fail = 0
    total_rib_mass = 0.0
    for i in range(len(rwb.y_positions)):
        y_mm = rwb.y_positions[i] * 1000
        h_mm = rwb.h_box_at_station[i] * 1000
        t_mm = rwb.t_rib_per_station[i] * 1000
        tau_r = rwb.tau_rib[i] / 1e6
        tau_c = rwb.tau_cr_rib[i] / 1e6
        mos = rwb.mos_rib[i]
        ok = rwb.pass_flags[i]

        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1

        mos_str = f"{mos:.2f}" if mos < 1000 else "INF"
        thickened = " *" if t_mm > rwb.t_rib_initial_mm + 0.01 else ""

        # Compute individual rib mass if geometry available
        mass_str = ""
        if show_mass and i < len(rib_geometries):
            rib_area = rib_geometries[i].S_total  # [m2]
            t_rib_m = rwb.t_rib_per_station[i]    # [m]
            mass_kg = rib_area * t_rib_m * rib_density
            mass_g = mass_kg * 1000
            total_rib_mass += mass_g
            mass_str = f"{mass_g:9.2f}"

        if show_mass:
            lines.append(
                f"{i:4d} {y_mm:8.1f} {h_mm:10.2f} {t_mm:10.1f}"
                f"{tau_r:13.4f} {tau_c:12.4f} {mos_str:>8} {mass_str} {status:>8}{thickened}"
            )
        else:
            lines.append(
                f"{i:4d} {y_mm:8.1f} {h_mm:10.2f} {t_mm:10.1f}"
                f"{tau_r:13.4f} {tau_c:12.4f} {mos_str:>8} {status:>8}{thickened}"
            )

    lines.append(sep)
    summary = (f"Summary: {n_pass} PASS, {n_fail} FAIL out of {len(rwb.y_positions)} ribs  |  "
               f"t_rib: {rwb.t_rib_initial_mm:.1f}mm initial, "
               f"{rwb.t_rib_max_mm:.1f}mm max  |  "
               f"{rwb.n_thickened} ribs thickened")
    if show_mass:
        summary += f"  |  Total rib mass: {total_rib_mass:.2f}g"
    lines.append(summary)
    if rwb.n_thickened > 0:
        lines.append("  (* = thickened from initial value)")
    lines.append(sep)

    return "\n".join(lines)


def compute_minimum_rib_count(AR: float, S_ref: float, c_MGC: float) -> int:
    """
    Compute minimum number of rib bays from geometric constraint.

    N_Rib_min = ceil(1 + sqrt(AR * S_ref) / c_MGC)

    This is consistent with the N_Rib formula used in main.py.

    Args:
        AR: Aspect ratio [-].
        S_ref: Reference wing area [m²].
        c_MGC: Mean geometric chord [m].

    Returns:
        Minimum number of rib bays [-].
    """
    return int(np.ceil(1 + np.sqrt(AR * S_ref) / c_MGC))


if __name__ == "__main__":
    from geometry import PlanformParams, SparPosition, compute_all_stations
    from materials import MaterialDatabase

    print("=== Ribs Module Test ===\n")

    # Setup
    planform = PlanformParams.from_input(
        b=2.0, AR=8.0, taper_ratio=0.5, t_c=0.12,
        S_ref=0.5, C_r_mm=300, c_MGC=0.25, Y_bar_mm=400
    )

    spar_pos = SparPosition(X_FS_percent=25, X_RS_percent=65)

    # Material
    db = MaterialDatabase()
    pla = db.get_material('PLA')

    # Rib properties
    rib_props = RibProperties.from_mm(t_rib_mm=2.0, material=pla)

    print(f"Rib Properties:")
    print(f"  t_rib = {rib_props.t_rib*1000:.1f} mm")
    print(f"  density = {rib_props.density} kg/m³")

    # Generate stations
    N_Rib = 5
    stations = compute_all_stations(planform, spar_pos, N_Rib)

    # Extract arrays
    y = np.array([s.y for s in stations])
    chord = np.array([s.chord for s in stations])
    h_box = np.array([s.h_box for s in stations])
    x_FS = np.array([s.x_FS for s in stations])
    x_RS = np.array([s.x_RS for s in stations])

    # Generate rib geometries
    ribs = generate_rib_geometries(y, chord, h_box, x_FS, x_RS)

    print(f"\nRib Geometries (N_Rib={N_Rib}):")
    print(f"  {'y[mm]':>8} {'chord[mm]':>10} {'S_LE_FS[mm²]':>14} {'S_FS_RS[mm²]':>14} {'S_total[mm²]':>14}")
    for rib in ribs:
        print(f"  {rib.y*1000:8.1f} {rib.chord*1000:10.1f} {rib.S_LE_FS*1e6:14.1f} {rib.S_FS_RS*1e6:14.1f} {rib.S_total*1e6:14.1f}")

    # Mass calculation
    mass_result = compute_total_rib_mass(ribs, rib_props, planform.L_span)

    print(f"\nRib Mass Summary:")
    print(f"  N_Rib = {mass_result.N_Rib}")
    print(f"  Spacing = {mass_result.spacing*1000:.1f} mm")
    print(f"  Total rib mass = {mass_result.total_rib_mass*1000:.2f} g")
    print(f"  Average rib mass = {mass_result.avg_rib_mass*1000:.2f} g")
    print(f"  Root rib area = {mass_result.root_rib_area*1e6:.1f} mm²")
    print(f"  Tip rib area = {mass_result.tip_rib_area*1e6:.1f} mm²")

    # Formatted outputs
    outputs = format_rib_outputs(ribs, rib_props, planform.L_span)
    print(f"\nFormatted Rib Outputs:")
    print(f"  N_Rib = {outputs.N_Rib}")
    print(f"  Spacing = {outputs.spacing_mm:.1f} mm")
    print(f"  t_rib = {outputs.t_rib_mm:.1f} mm")
    print(f"  S_(Rib LE-FS) = {outputs.S_Rib_LE_FS_mm2:.1f} mm²")
    print(f"  S_(Rib FS-RS) = {outputs.S_Rib_FS_RS_mm2:.1f} mm²")
    print(f"  S_Rib = {outputs.S_Rib_total_mm2:.1f} mm²")
    print(f"  Total mass = {outputs.total_mass_g:.2f} g")

    # Simple mass calculation comparison
    S_avg = (ribs[0].S_total + ribs[-1].S_total) / 2
    m_simple = compute_rib_mass_simple(N_Rib, S_avg, rib_props.t_rib, rib_props.density)
    print(f"\nSimple mass calc (avg area): {m_simple*1000:.2f} g")

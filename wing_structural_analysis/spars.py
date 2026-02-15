"""
spars.py - Spar Structural Analysis Module

Handles spar bending stress, shear stress, von Mises stress,
load sharing between spars, and inertia calculations.

All internal calculations use SI units (N, m, Pa, m⁴).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class SparProperties:
    """Spar cross-sectional properties (circular tube)."""
    d_outer: float      # Outer diameter [m]
    t_wall: float       # Wall thickness [m]
    E: float            # Young's modulus [Pa]
    G: float            # Shear modulus [Pa]
    sigma_u: float      # Ultimate tensile strength [Pa]
    tau_u: float        # Ultimate shear strength [Pa]
    density: float      # Material density [kg/m³]

    @property
    def d_inner(self) -> float:
        """Inner diameter [m]."""
        return self.d_outer - 2 * self.t_wall

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return (math.pi / 4) * (self.d_outer**2 - self.d_inner**2)

    @property
    def I(self) -> float:
        """Second moment of area (bending inertia) [m⁴]."""
        return (math.pi / 64) * (self.d_outer**4 - self.d_inner**4)

    @property
    def c_dist(self) -> float:
        """Distance from neutral axis to outer fiber [m]."""
        return self.d_outer / 2

    @property
    def J(self) -> float:
        """Polar moment of inertia [m⁴]."""
        # J = (π/32) × (d_o⁴ - d_i⁴) = 2 × I
        return (math.pi / 32) * (self.d_outer**4 - self.d_inner**4)

    @property
    def A_shear(self) -> float:
        """Effective shear area (approximation for tube) [m²]."""
        # For thin-walled tube, shear area ≈ 0.5 * total area
        # Conservative approximation: use full area
        return self.area

    def validate(self) -> Tuple[bool, str]:
        """
        Validate spar geometry.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.d_outer <= 0:
            return False, "Outer diameter must be positive"
        if self.t_wall <= 0:
            return False, "Wall thickness must be positive"
        if self.t_wall >= self.d_outer / 2:
            return False, "Wall thickness must be less than radius"
        return True, ""

    @classmethod
    def from_mm(cls, d_outer_mm: float, t_wall_mm: float,
                material) -> 'SparProperties':
        """
        Create from millimeter inputs and material object.

        Args:
            d_outer_mm: Outer diameter [mm]
            t_wall_mm: Wall thickness [mm]
            material: Material object with E, G, sigma_u, tau_u, density

        Returns:
            SparProperties instance
        """
        return cls(
            d_outer=d_outer_mm / 1000,
            t_wall=t_wall_mm / 1000,
            E=material.E,
            G=material.G,
            sigma_u=material.sigma_u,
            tau_u=material.tau_u,
            density=material.density
        )


@dataclass
class LoadSharing:
    """Load sharing between front and rear spars."""
    eta_FS: float   # Front spar share [-]
    eta_RS: float   # Rear spar share [-]

    @classmethod
    def from_positions(cls, X_FS_percent: float, X_RS_percent: float, x_ac_percent: float) -> 'LoadSharing':
        """
        Calculate load sharing based on spar positions and aerodynamic center.

        η_FS = (X_RS% - x_ac) / (X_RS% - X_FS%)
        η_RS = (x_ac - X_FS%) / (X_RS% - X_FS%)

        Args:
            X_FS_percent: Front spar position [% chord]
            X_RS_percent: Rear spar position [% chord]
            x_ac_percent: Aerodynamic center position [% chord]

        Returns:
            LoadSharing instance
        """
        span = X_RS_percent - X_FS_percent
        if span <= 0:
            raise ValueError("X_RS must be greater than X_FS")

        eta_FS = (X_RS_percent - x_ac_percent) / span
        eta_RS = (x_ac_percent - X_FS_percent) / span

        # Clamp to valid range [0, 1]
        eta_FS = max(0.0, min(1.0, eta_FS))
        eta_RS = max(0.0, min(1.0, eta_RS))

        return cls(eta_FS=eta_FS, eta_RS=eta_RS)

    @classmethod
    def from_inertias(cls, I_FS: float, I_RS: float) -> 'LoadSharing':
        """
        Calculate load sharing based on bending inertias (legacy method).

        η_FS = I_FS / (I_FS + I_RS)
        η_RS = I_RS / (I_FS + I_RS)

        Args:
            I_FS: Front spar inertia [m⁴]
            I_RS: Rear spar inertia [m⁴]

        Returns:
            LoadSharing instance
        """
        I_total = I_FS + I_RS
        if I_total <= 0:
            raise ValueError("Total inertia must be positive")

        return cls(
            eta_FS=I_FS / I_total,
            eta_RS=I_RS / I_total
        )


# =============================================================================
# STRESS CALCULATIONS
# =============================================================================

def bending_stress(M: float, I: float, c: float) -> float:
    """
    Calculate bending stress at outer fiber.

    σ_b = M * c / I

    Args:
        M: Bending moment [N·m]
        I: Second moment of area [m⁴]
        c: Distance to outer fiber [m]

    Returns:
        Bending stress [Pa]
    """
    if I <= 0:
        raise ValueError("Inertia must be positive")
    return M * c / I


def torsional_shear_stress(T: float, r: float, J: float) -> float:
    """
    Calculate torsional shear stress in circular spar.

    τ = T × r / J

    Args:
        T: Torsion moment [N·m]
        r: Outer radius [m]
        J: Polar moment of inertia [m⁴]

    Returns:
        Shear stress [Pa]
    """
    if J <= 0:
        raise ValueError("Polar moment must be positive")
    return T * r / J


def von_mises_stress(sigma: float, tau: float) -> float:
    """
    Calculate von Mises equivalent stress.

    σ_vm = √(σ² + 3τ²)

    Args:
        sigma: Normal stress [Pa]
        tau: Shear stress [Pa]

    Returns:
        Von Mises stress [Pa]
    """
    return math.sqrt(sigma**2 + 3 * tau**2)


@dataclass
class SparStressResult:
    """Stress analysis result for a spar at a station."""
    y: float            # Spanwise position [m]
    M_share: float      # Shared bending moment [N·m]
    T_share: float      # Shared torsion moment [N·m]
    sigma_b: float      # Bending stress [Pa]
    tau: float          # Torsional shear stress [Pa]
    sigma_vm: float     # Von Mises stress [Pa]
    eta: float          # Load share [-]


def analyze_spar_stress(M: float, T: float, spar: SparProperties,
                        eta: float, y: float = 0.0) -> SparStressResult:
    """
    Analyze stress in a spar at a station.

    Args:
        M: Total bending moment at station [N·m]
        T: Total torsion moment at station [N·m]
        spar: Spar properties
        eta: Load share for this spar [-]
        y: Spanwise position [m]

    Returns:
        SparStressResult
    """
    # Shared loads
    M_share = eta * M
    T_share = eta * T

    # Stresses
    sigma_b = bending_stress(M_share, spar.I, spar.c_dist)
    tau = torsional_shear_stress(T_share, spar.c_dist, spar.J)
    sigma_vm = von_mises_stress(sigma_b, tau)

    return SparStressResult(
        y=y,
        M_share=M_share,
        T_share=T_share,
        sigma_b=sigma_b,
        tau=tau,
        sigma_vm=sigma_vm,
        eta=eta
    )


# =============================================================================
# DUAL SPAR ANALYSIS
# =============================================================================

@dataclass
class DualSparResult:
    """Combined result for front and rear spars."""
    FS: SparStressResult
    RS: SparStressResult
    load_sharing: LoadSharing

    @property
    def max_sigma_vm(self) -> float:
        """Maximum von Mises stress between both spars."""
        return max(abs(self.FS.sigma_vm), abs(self.RS.sigma_vm))

    @property
    def critical_spar(self) -> str:
        """Which spar is critical."""
        if abs(self.FS.sigma_vm) >= abs(self.RS.sigma_vm):
            return "FS"
        return "RS"


def analyze_dual_spars(M: float, T: float,
                       spar_FS: SparProperties,
                       spar_RS: SparProperties,
                       X_FS_percent: float,
                       X_RS_percent: float,
                       x_ac_percent: float,
                       y: float = 0.0) -> DualSparResult:
    """
    Analyze stress in both spars at a station.

    Args:
        M: Total bending moment [N·m]
        T: Total torsion moment [N·m]
        spar_FS: Front spar properties
        spar_RS: Rear spar properties
        X_FS_percent: Front spar position [% chord]
        X_RS_percent: Rear spar position [% chord]
        x_ac_percent: Aerodynamic center position [% chord]
        y: Spanwise position [m]

    Returns:
        DualSparResult
    """
    # Load sharing based on positions
    sharing = LoadSharing.from_positions(X_FS_percent, X_RS_percent, x_ac_percent)

    # Analyze each spar
    FS_result = analyze_spar_stress(M, T, spar_FS, sharing.eta_FS, y)
    RS_result = analyze_spar_stress(M, T, spar_RS, sharing.eta_RS, y)

    return DualSparResult(
        FS=FS_result,
        RS=RS_result,
        load_sharing=sharing
    )


def analyze_spars_all_stations(y: np.ndarray, M: np.ndarray, T: np.ndarray,
                                spar_FS: SparProperties,
                                spar_RS: SparProperties,
                                X_FS_percent: float,
                                X_RS_percent: float,
                                x_ac_percent: float) -> list:
    """
    Analyze both spars at all stations.

    Args:
        y: Spanwise positions [m]
        M: Bending moment at each station [N·m]
        T: Torsion moment at each station [N·m]
        spar_FS: Front spar properties
        spar_RS: Rear spar properties
        X_FS_percent: Front spar position [% chord]
        X_RS_percent: Rear spar position [% chord]
        x_ac_percent: Aerodynamic center position [% chord]

    Returns:
        List of DualSparResult for each station
    """
    results = []
    for i in range(len(y)):
        result = analyze_dual_spars(M[i], T[i], spar_FS, spar_RS,
                                    X_FS_percent, X_RS_percent, x_ac_percent, y[i])
        results.append(result)
    return results


def find_critical_spar_station(results: list) -> Tuple[int, DualSparResult, str]:
    """
    Find the station with maximum spar stress.

    Args:
        results: List of DualSparResult

    Returns:
        Tuple of (index, result, which_spar)
    """
    max_idx = 0
    max_vm = results[0].max_sigma_vm

    for i, r in enumerate(results):
        if r.max_sigma_vm > max_vm:
            max_vm = r.max_sigma_vm
            max_idx = i

    return max_idx, results[max_idx], results[max_idx].critical_spar


# =============================================================================
# SPAR MASS CALCULATION
# =============================================================================

def spar_mass(spar: SparProperties, length: float) -> float:
    """
    Calculate spar mass.

    m = A * L * ρ

    Args:
        spar: Spar properties
        length: Spar length [m]

    Returns:
        Mass [kg]
    """
    return spar.area * length * spar.density


# =============================================================================
# CRITICAL AREA (Actual vs Required)
# =============================================================================

def critical_area_bending(M: float, sigma_allow: float, c_dist: float,
                          shape_factor: float = 1.0) -> float:
    """
    Estimate critical (minimum required) area for bending.

    For a tube with known c/r ratio, A_crit ≈ M * c / (σ_allow * I/A)

    This is a rough estimate. For circular tube:
    I/A ≈ (d_o² + d_i²) / 16 ≈ d_o² / 8 for thin wall

    Simplified: A_crit ≈ 4 * M / (σ_allow * d_o)

    Args:
        M: Bending moment [N·m]
        sigma_allow: Allowable stress [Pa]
        c_dist: Distance to outer fiber [m] (≈ d_o/2)
        shape_factor: Correction factor for tube geometry

    Returns:
        Critical area [m²]
    """
    # For thin tube: I ≈ π * r³ * t, A ≈ 2 * π * r * t
    # I/A ≈ r²/2 = (d_o/2)²/2 = d_o²/8
    # σ = M * c / I = M * (d_o/2) / (A * d_o²/8) = 4M / (A * d_o)
    # A = 4M / (σ * d_o) = 4M / (σ * 2c) = 2M / (σ * c)

    return shape_factor * 2 * abs(M) / (sigma_allow * c_dist)


# =============================================================================
# DEFLECTION ESTIMATE
# =============================================================================

def tip_deflection_uniform_load(w: float, L: float, E: float, I: float) -> float:
    """
    Estimate tip deflection for uniformly loaded cantilever.

    δ = w * L⁴ / (8 * E * I)

    Args:
        w: Distributed load [N/m]
        L: Span length [m]
        E: Equivalent modulus [Pa]
        I: Equivalent inertia [m⁴]

    Returns:
        Tip deflection [m]
    """
    return w * L**4 / (8 * E * I)


def tip_deflection_point_load(P: float, L: float, E: float, I: float) -> float:
    """
    Tip deflection for point load at tip.

    δ = P * L³ / (3 * E * I)

    Args:
        P: Point load [N]
        L: Span length [m]
        E: Equivalent modulus [Pa]
        I: Equivalent inertia [m⁴]

    Returns:
        Tip deflection [m]
    """
    return P * L**3 / (3 * E * I)


def tip_deflection_from_moment(y: np.ndarray, M: np.ndarray, E: float, I: float) -> float:
    """
    Calculate tip deflection using numerical integration of M(y).

    Uses moment-area method:
    δ_tip = ∫₀ᴸ (L-y) * M(y) / (E*I) dy

    This works for ANY load distribution (uniform, elliptic, etc.)

    Args:
        y: Spanwise positions [m] (from root to tip)
        M: Bending moment at each station [N·m]
        E: Young's modulus [Pa]
        I: Second moment of area [m⁴]

    Returns:
        Tip deflection [m]
    """
    if E <= 0 or I <= 0:
        raise ValueError("E and I must be positive")

    L_span = y[-1]
    # Integrand: (L - y) * M(y) / (E * I)
    integrand = (L_span - y) * M / (E * I)

    # Trapezoidal integration
    delta_tip = np.trapz(integrand, y)

    return delta_tip


# =============================================================================
# BOX BEAM MODEL (Transformed Section Method)
# =============================================================================

@dataclass
class BoxBeamSection:
    """Wing box beam cross-section using transformed section method.

    Reference material is spar material (E_ref = E_spar).
    Skin contribution is transformed by modular ratio n = E_skin / E_spar.
    Spar tubes are assumed centered on the neutral axis (mid-height of box).
    """
    I_total: float      # Total second moment of area [m⁴] (ref material)
    E_ref: float        # Reference modulus (spar) [Pa]
    n_skin: float       # Modular ratio E_skin/E_ref [-]
    I_FS: float         # Front spar I contribution [m⁴]
    I_RS: float         # Rear spar I contribution [m⁴]
    I_skin: float       # Transformed skin I contribution [m⁴]
    h_box: float        # Box height [m]
    w_box: float        # Box width between spars [m]
    t_skin: float       # Skin thickness [m]

    @property
    def EI(self) -> float:
        """Bending stiffness [N·m²]."""
        return self.E_ref * self.I_total

    @property
    def skin_fraction(self) -> float:
        """Fraction of I contributed by skin [-]."""
        return self.I_skin / self.I_total if self.I_total > 0 else 0.0


def compute_box_beam_section(spar_FS: SparProperties, spar_RS: SparProperties,
                              h_box: float, w_box: float,
                              t_skin: float, E_skin: float) -> BoxBeamSection:
    """
    Compute transformed section properties for wing box beam.

    Uses parallel axis theorem for skin panels at ±h_box/2 from NA.
    Spar tubes centered on NA contribute only their own I.

    Args:
        spar_FS: Front spar properties
        spar_RS: Rear spar properties
        h_box: Box height [m]
        w_box: Box width between spars [m]
        t_skin: Skin thickness [m]
        E_skin: Skin Young's modulus [Pa]

    Returns:
        BoxBeamSection with transformed section properties
    """
    E_ref = spar_FS.E
    n = E_skin / E_ref  # Modular ratio

    # Upper + lower skin panels via parallel axis theorem
    # Each panel: w_box * t_skin at distance h_box/2 from NA
    d = h_box / 2
    I_skin_parallel = 2 * w_box * t_skin * d**2    # Steiner term (dominant)
    I_skin_self = 2 * w_box * t_skin**3 / 12       # Self-inertia (small)
    I_skin_transformed = n * (I_skin_parallel + I_skin_self)

    I_total = spar_FS.I + spar_RS.I + I_skin_transformed

    return BoxBeamSection(
        I_total=I_total,
        E_ref=E_ref,
        n_skin=n,
        I_FS=spar_FS.I,
        I_RS=spar_RS.I,
        I_skin=I_skin_transformed,
        h_box=h_box,
        w_box=w_box,
        t_skin=t_skin
    )


@dataclass
class BoxBeamStressResult:
    """Box beam stress analysis result at a station."""
    y: float                 # Spanwise position [m]
    sigma_b_FS: float        # Bending stress at FS outer fiber [Pa]
    sigma_b_RS: float        # Bending stress at RS outer fiber [Pa]
    sigma_skin_comp: float   # Skin compression stress (actual, in skin material) [Pa]
    tau_FS: float            # Torsional shear stress in FS [Pa]
    tau_RS: float            # Torsional shear stress in RS [Pa]
    sigma_vm_FS: float       # Von Mises stress FS [Pa]
    sigma_vm_RS: float       # Von Mises stress RS [Pa]
    I_box: float             # Box beam I used [m⁴]


def analyze_box_beam_stress(M: float, T: float,
                            box: BoxBeamSection,
                            spar_FS: SparProperties,
                            spar_RS: SparProperties,
                            eta_FS: float, eta_RS: float,
                            y: float = 0.0) -> BoxBeamStressResult:
    """
    Analyze stresses using wing box beam model.

    Bending: Full wing box I resists the entire moment M.
    Torsion: Split between spar tubes by load sharing (position-based).

    Args:
        M: Total bending moment at station [N·m]
        T: Total torsion moment at station [N·m]
        box: Box beam section properties
        spar_FS: Front spar properties
        spar_RS: Rear spar properties
        eta_FS: Front spar torsion share [-]
        eta_RS: Rear spar torsion share [-]
        y: Spanwise position [m]

    Returns:
        BoxBeamStressResult
    """
    I = box.I_total

    # Bending stresses (entire M resisted by full box section)
    sigma_b_FS = abs(M) * spar_FS.c_dist / I
    sigma_b_RS = abs(M) * spar_RS.c_dist / I

    # Skin bending stress (actual stress in skin material)
    # sigma_ref = M * (h/2) / I, then actual = n * sigma_ref
    sigma_skin = box.n_skin * abs(M) * (box.h_box / 2) / I

    # Torsional shear in spar tubes (split by load sharing)
    tau_FS_val = abs(T * eta_FS) * spar_FS.c_dist / spar_FS.J
    tau_RS_val = abs(T * eta_RS) * spar_RS.c_dist / spar_RS.J

    return BoxBeamStressResult(
        y=y,
        sigma_b_FS=sigma_b_FS,
        sigma_b_RS=sigma_b_RS,
        sigma_skin_comp=sigma_skin,
        tau_FS=tau_FS_val,
        tau_RS=tau_RS_val,
        sigma_vm_FS=von_mises_stress(sigma_b_FS, tau_FS_val),
        sigma_vm_RS=von_mises_stress(sigma_b_RS, tau_RS_val),
        I_box=I
    )


def analyze_box_beam_all_stations(y: np.ndarray, M: np.ndarray, T: np.ndarray,
                                   box_sections: list,
                                   spar_FS: SparProperties,
                                   spar_RS: SparProperties,
                                   eta_FS: float, eta_RS: float) -> list:
    """
    Analyze box beam stresses at all stations.

    Args:
        y: Spanwise positions [m]
        M: Bending moment at each station [N·m]
        T: Torsion at each station [N·m]
        box_sections: BoxBeamSection at each station
        spar_FS: Front spar properties
        spar_RS: Rear spar properties
        eta_FS: Front spar torsion share [-]
        eta_RS: Rear spar torsion share [-]

    Returns:
        List of BoxBeamStressResult
    """
    return [analyze_box_beam_stress(M[i], T[i], box_sections[i],
                                     spar_FS, spar_RS, eta_FS, eta_RS, y[i])
            for i in range(len(y))]


def find_critical_box_beam_station(results: list) -> Tuple[int, BoxBeamStressResult]:
    """
    Find station with maximum von Mises stress (box beam model).

    Returns:
        Tuple of (index, critical_result)
    """
    max_idx = 0
    max_vm = max(results[0].sigma_vm_FS, results[0].sigma_vm_RS)

    for i, r in enumerate(results):
        vm = max(r.sigma_vm_FS, r.sigma_vm_RS)
        if vm > max_vm:
            max_vm = vm
            max_idx = i

    return max_idx, results[max_idx]


if __name__ == "__main__":
    from materials import MaterialDatabase

    print("=== Spars Module Test ===\n")

    # Material
    db = MaterialDatabase()
    al = db.get_material('AL7075-T6')

    # Create spars
    spar_FS = SparProperties.from_mm(d_outer_mm=20, t_wall_mm=2, material=al)
    spar_RS = SparProperties.from_mm(d_outer_mm=16, t_wall_mm=1.5, material=al)

    print("Front Spar:")
    print(f"  d_outer = {spar_FS.d_outer*1000:.1f} mm")
    print(f"  t_wall = {spar_FS.t_wall*1000:.1f} mm")
    print(f"  Area = {spar_FS.area*1e6:.2f} mm²")
    print(f"  I = {spar_FS.I*1e12:.2f} mm⁴")
    valid, msg = spar_FS.validate()
    print(f"  Valid: {valid}")

    print("\nRear Spar:")
    print(f"  d_outer = {spar_RS.d_outer*1000:.1f} mm")
    print(f"  t_wall = {spar_RS.t_wall*1000:.1f} mm")
    print(f"  Area = {spar_RS.area*1e6:.2f} mm²")
    print(f"  I = {spar_RS.I*1e12:.2f} mm⁴")

    # Load sharing
    sharing = LoadSharing.from_inertias(spar_FS.I, spar_RS.I)
    print(f"\nLoad Sharing:")
    print(f"  η_FS = {sharing.eta_FS:.3f} ({sharing.eta_FS*100:.1f}%)")
    print(f"  η_RS = {sharing.eta_RS:.3f} ({sharing.eta_RS*100:.1f}%)")

    # Stress analysis at root
    M_root = 20.0  # N·m
    T_root = 0.5   # N·m (torsion)

    print(f"\n--- Stress Analysis at Root ---")
    print(f"M = {M_root} N·m, T = {T_root} N·m")

    result = analyze_dual_spars(M_root, T_root, spar_FS, spar_RS,
                                X_FS_percent=25.0, X_RS_percent=75.0, x_ac_percent=25.0)

    print(f"\nFront Spar:")
    print(f"  M_share = {result.FS.M_share:.2f} N·m")
    print(f"  σ_b = {result.FS.sigma_b/1e6:.2f} MPa")
    print(f"  τ = {result.FS.tau/1e6:.4f} MPa")
    print(f"  σ_vm = {result.FS.sigma_vm/1e6:.2f} MPa")

    print(f"\nRear Spar:")
    print(f"  M_share = {result.RS.M_share:.2f} N·m")
    print(f"  σ_b = {result.RS.sigma_b/1e6:.2f} MPa")
    print(f"  τ = {result.RS.tau/1e6:.4f} MPa")
    print(f"  σ_vm = {result.RS.sigma_vm/1e6:.2f} MPa")

    print(f"\nCritical spar: {result.critical_spar}")
    print(f"Max σ_vm = {result.max_sigma_vm/1e6:.2f} MPa")

    # Check against allowable
    SF = 1.5
    sigma_allow = al.sigma_u / SF
    margin = (sigma_allow - result.max_sigma_vm) / sigma_allow * 100
    print(f"\nAllowable σ (SF={SF}): {sigma_allow/1e6:.1f} MPa")
    print(f"Safety margin: {margin:.1f}%")

    # Mass calculation
    L_span = 1.0  # m
    m_FS = spar_mass(spar_FS, L_span)
    m_RS = spar_mass(spar_RS, L_span)
    print(f"\n--- Spar Masses (L={L_span}m) ---")
    print(f"  m_FS = {m_FS*1000:.1f} g")
    print(f"  m_RS = {m_RS*1000:.1f} g")
    print(f"  Total = {(m_FS+m_RS)*1000:.1f} g")

    # Deflection estimate
    w_avg = 50  # N/m
    I_total = spar_FS.I + spar_RS.I
    E = al.E
    delta = tip_deflection_uniform_load(w_avg, L_span, E, I_total)
    print(f"\n--- Tip Deflection Estimate ---")
    print(f"  δ_tip = {delta*1000:.2f} mm")
    print(f"  δ/L = {delta/L_span*100:.2f}%")
    print(f"  Limit (L/20) = {L_span/20*1000:.1f} mm")

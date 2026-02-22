"""
geometry.py - Planform and Wing-Box Geometry Module

Handles wing planform calculations, rib stations, spar positions,
and wing-box cross-section properties.

All internal calculations use SI units (m, m²).
User interface accepts mm where noted.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


@dataclass
class SectionGeometry:
    """Geometry for one taper section of the wing."""
    c_root: float   # Root chord of this section [m]
    c_tip: float    # Tip chord of this section [m]
    b_section: float  # Span of this section [m]


@dataclass
class PlanformParams:
    """Wing planform parameters."""
    b: float            # Full wingspan [m]
    AR: float           # Aspect ratio [-]
    taper_ratio: float  # λ, taper ratio [-] (tip chord / root chord)
    t_c: float          # Thickness-to-chord ratio [-]
    S_ref: float        # Reference wing area [m²]
    C_r: float          # Root chord [m]
    c_MGC: float        # Mean geometric chord [m]
    Y_bar: float        # y distance from root to MGC [m]
    sections: Optional[List[SectionGeometry]] = field(default=None, repr=False)

    @property
    def is_multi_section(self) -> bool:
        """True if wing has multiple taper sections."""
        return self.sections is not None and len(self.sections) > 1

    @property
    def L_span(self) -> float:
        """Half-wing span [m]."""
        return self.b / 2

    @property
    def c_tip(self) -> float:
        """Tip chord [m]."""
        return self.taper_ratio * self.C_r

    @classmethod
    def from_input(cls, b: float, AR: float, taper_ratio: float,
                   t_c: float, S_ref: float, C_r_mm: float,
                   c_MGC: float, Y_bar_mm: float) -> 'PlanformParams':
        """
        Create from user input (with unit conversions).

        Args:
            b: Wingspan [m]
            AR: Aspect ratio [-]
            taper_ratio: Taper ratio [-]
            t_c: Thickness-to-chord ratio [-]
            S_ref: Reference area [m²]
            C_r_mm: Root chord [mm]
            c_MGC: Mean geometric chord [m]
            Y_bar_mm: Y distance to MGC [mm]

        Returns:
            PlanformParams instance
        """
        return cls(
            b=b,
            AR=AR,
            taper_ratio=taper_ratio,
            t_c=t_c,
            S_ref=S_ref,
            C_r=C_r_mm / 1000,  # mm to m
            c_MGC=c_MGC,
            Y_bar=Y_bar_mm / 1000  # mm to m
        )


def chord_at_station(y: float, C_r: float, C_tip: float, L_span: float) -> float:
    """
    Calculate chord at spanwise station y (linear taper).

    c(y) = c_root - (c_root - c_tip) * (y / L_span)

    Args:
        y: Spanwise position from root [m]
        C_r: Root chord [m]
        C_tip: Tip chord [m]
        L_span: Half-span [m]

    Returns:
        Chord at station y [m]
    """
    return C_r - (C_r - C_tip) * (y / L_span)


def chord_at_station_multi(y: float, sections: List[SectionGeometry]) -> float:
    """
    Calculate chord at spanwise station y for a multi-section wing.

    Each section is a trapezoidal segment with its own root/tip chord
    and span.  Sections are concatenated root-to-tip.

    Args:
        y: Spanwise position from root [m]
        sections: List of SectionGeometry (ordered root→tip)

    Returns:
        Chord at station y [m]
    """
    y_start = 0.0
    for sec in sections:
        y_end = y_start + sec.b_section
        if y <= y_end or sec is sections[-1]:
            eta = (y - y_start) / sec.b_section if sec.b_section > 0 else 0.0
            eta = max(0.0, min(1.0, eta))
            return sec.c_root + (sec.c_tip - sec.c_root) * eta
        y_start = y_end
    # Fallback (should not reach here)
    return sections[-1].c_tip


def generate_rib_stations(N_Rib: int, L_span: float) -> np.ndarray:
    """
    Generate rib station positions along half-span.

    y_i = i * (L_span / N_Rib), i = 0..N_Rib

    Args:
        N_Rib: Number of rib bays (N_Rib+1 stations including root)
        L_span: Half-span [m]

    Returns:
        Array of y positions [m]
    """
    return np.linspace(0, L_span, N_Rib + 1)


@dataclass
class SparPosition:
    """Spar position as percentage of chord."""
    X_FS_percent: float  # Front spar position [% chord]
    X_RS_percent: float  # Rear spar position [% chord]

    def validate(self) -> bool:
        """Check that front spar is ahead of rear spar."""
        return self.X_FS_percent < self.X_RS_percent

    def x_FS_at_station(self, chord: float) -> float:
        """Front spar x-position at given chord [m]."""
        return (self.X_FS_percent / 100) * chord

    def x_RS_at_station(self, chord: float) -> float:
        """Rear spar x-position at given chord [m]."""
        return (self.X_RS_percent / 100) * chord


@dataclass
class SparGeometry:
    """Spar cross-section geometry (circular tube)."""
    d_outer: float  # Outer diameter [m]
    t_wall: float   # Wall thickness [m]

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

    def validate(self) -> bool:
        """Check geometric validity."""
        return (self.d_outer > 0 and
                self.t_wall > 0 and
                self.t_wall < self.d_outer / 2)

    @classmethod
    def from_mm(cls, d_outer_mm: float, t_wall_mm: float) -> 'SparGeometry':
        """Create from millimeter inputs."""
        return cls(
            d_outer=d_outer_mm / 1000,
            t_wall=t_wall_mm / 1000
        )


@dataclass
class WingBoxSection:
    """Wing-box cross-section at a station."""
    y: float            # Spanwise position [m]
    chord: float        # Local chord [m]
    x_FS: float         # Front spar x-position [m]
    x_RS: float         # Rear spar x-position [m]
    h_box: float        # Box height [m]
    A_m: float          # Median (enclosed) area [m²]

    @property
    def width(self) -> float:
        """Box width (distance between spars) [m]."""
        return self.x_RS - self.x_FS


def compute_wing_box_section(y: float, planform: PlanformParams,
                             spar_pos: SparPosition) -> WingBoxSection:
    """
    Compute wing-box cross-section properties at station y.

    Uses multi-section chord interpolation when planform has multiple
    taper sections, otherwise falls back to single linear taper.

    Args:
        y: Spanwise position [m]
        planform: Planform parameters
        spar_pos: Spar position percentages

    Returns:
        WingBoxSection with computed properties
    """
    if planform.is_multi_section:
        chord = chord_at_station_multi(y, planform.sections)
    else:
        chord = chord_at_station(y, planform.C_r, planform.c_tip, planform.L_span)
    x_FS = spar_pos.x_FS_at_station(chord)
    x_RS = spar_pos.x_RS_at_station(chord)
    h_box = planform.t_c * chord
    A_m = (x_RS - x_FS) * h_box  # Rectangular approximation

    return WingBoxSection(
        y=y,
        chord=chord,
        x_FS=x_FS,
        x_RS=x_RS,
        h_box=h_box,
        A_m=A_m
    )


def compute_all_stations(planform: PlanformParams,
                         spar_pos: SparPosition,
                         N_Rib: int) -> List[WingBoxSection]:
    """
    Compute wing-box sections at all rib stations.

    Args:
        planform: Planform parameters
        spar_pos: Spar position percentages
        N_Rib: Number of rib bays

    Returns:
        List of WingBoxSection for each station
    """
    y_stations = generate_rib_stations(N_Rib, planform.L_span)
    return [compute_wing_box_section(y, planform, spar_pos) for y in y_stations]


def compute_spar_sweep_angle(spar_pos_percent: float, planform: PlanformParams,
                              Lambda_ac_deg: float = 0.0, x_ac_percent: float = 25.0) -> float:
    """
    Compute sweep angle of spar using aerodynamic formula.

    Λ_spar = arctan(tan(Λ_ac) + (4×(x_ac - X_spar%)×(1-λ)) / (x_ac×AR×(1+λ)))

    Args:
        spar_pos_percent: Spar position [% chord]
        planform: Planform parameters
        Lambda_ac_deg: Aerodynamic center sweep angle [degrees]
        x_ac_percent: Aerodynamic center position [% chord]

    Returns:
        Sweep angle [degrees]
    """
    # Convert to radians
    Lambda_ac_rad = math.radians(Lambda_ac_deg)

    # Get taper ratio and aspect ratio
    lambda_taper = planform.taper_ratio
    AR = planform.AR

    # Avoid division by zero
    if x_ac_percent <= 0:
        x_ac_percent = 25.0  # default

    # Formula: Λ = arctan(tan(Λ_ac) + (4×(x_ac - X_spar%)×(1-λ)) / (x_ac×AR×(1+λ)))
    term1 = math.tan(Lambda_ac_rad)
    term2 = (4 * (x_ac_percent - spar_pos_percent) * (1 - lambda_taper)) / (x_ac_percent * AR * (1 + lambda_taper))

    sweep_rad = math.atan(term1 + term2)
    return math.degrees(sweep_rad)


@dataclass
class SkinArcLengths:
    """Skin arc lengths at root (preliminary approximation)."""
    L_LE_FS: float   # Leading edge to front spar [m]
    L_LE_RS: float   # Leading edge to rear spar [m]
    L_FS_RS: float   # Front spar to rear spar [m]


def compute_skin_arc_lengths_root(planform: PlanformParams,
                                  spar_pos: SparPosition) -> SkinArcLengths:
    """
    Compute skin arc lengths at root using parabolic airfoil approximation.

    Formulas:
    L_(LE-FS) = √((h/2)² + (2×X_FS)²) + ((h/2)² / (4×λ×X_FS)) × asinh((4×X_FS)/h)
    L_(LE-RS) = √((h/2)² + (2×X_RS)²) + ((h/2)² / (4×λ×X_RS)) × asinh((4×X_RS)/h)
    L_(FS-RS) = L_(LE-RS) - L_(LE-FS)

    Args:
        planform: Planform parameters
        spar_pos: Spar position percentages

    Returns:
        SkinArcLengths at root
    """
    # Get positions in meters
    X_FS = spar_pos.x_FS_at_station(planform.C_r)  # [m]
    X_RS = spar_pos.x_RS_at_station(planform.C_r)  # [m]

    # Wing box height at root (h_FS = h_RS = h_box)
    h = planform.t_c * planform.C_r  # [m]

    # Taper ratio
    lambda_taper = planform.taper_ratio
    if lambda_taper <= 0:
        lambda_taper = 0.01  # Avoid division by zero

    # Parabolic arc length formula with taper ratio
    def arc_length_LE_to_spar(X: float, h: float, lam: float) -> float:
        """Calculate arc length from LE to spar position."""
        if X <= 0 or h <= 0:
            return 0.0
        h_half = h / 2
        term1 = math.sqrt(h_half**2 + (2 * X)**2)
        term2 = (h_half**2 / (4 * lam * X)) * math.asinh((4 * X) / h)
        return term1 + term2

    L_LE_FS = arc_length_LE_to_spar(X_FS, h, lambda_taper)
    L_LE_RS = arc_length_LE_to_spar(X_RS, h, lambda_taper)
    L_FS_RS = L_LE_RS - L_LE_FS

    return SkinArcLengths(
        L_LE_FS=L_LE_FS,
        L_LE_RS=L_LE_RS,
        L_FS_RS=L_FS_RS
    )


def compute_skin_area_half_wing(planform: PlanformParams) -> float:
    """
    Compute approximate skin wetted area for half-wing.

    S_skin_half ≈ 2 * ∫_0^{L_span} c(y) dy
    (factor 2 for upper + lower surface)

    For linear taper:
    ∫ c(y) dy = c_root * L - (c_root - c_tip) * L²/(2*L_span)
              = L_span * (c_root + c_tip) / 2

    Args:
        planform: Planform parameters

    Returns:
        Half-wing skin area [m²]
    """
    avg_chord = (planform.C_r + planform.c_tip) / 2
    return 2 * avg_chord * planform.L_span


def compute_rib_areas_root(planform: PlanformParams,
                           spar_pos: SparPosition) -> Tuple[float, float, float]:
    """
    Compute rib panel areas at root using parabolic airfoil approximation.

    Formulas (parabolic profile):
    S_Rib_LE_FS = 2 × X_FS × h_FS / 3
    S_Rib_FS_RS = (2 × X_RS × h_RS / 3) - S_Rib_LE_FS

    Args:
        planform: Planform parameters
        spar_pos: Spar position percentages

    Returns:
        Tuple of (S_Rib_LE_FS, S_Rib_FS_RS, S_Rib_total) [m²]
    """
    h_box_root = planform.t_c * planform.C_r  # h_FS = h_RS = h_box
    x_FS_root = spar_pos.x_FS_at_station(planform.C_r)
    x_RS_root = spar_pos.x_RS_at_station(planform.C_r)

    # Parabolic profile: area = (2/3) × base × height
    S_Rib_LE_FS = (2 / 3) * x_FS_root * h_box_root
    S_Rib_LE_RS = (2 / 3) * x_RS_root * h_box_root
    S_Rib_FS_RS = S_Rib_LE_RS - S_Rib_LE_FS
    S_Rib_total = S_Rib_LE_FS + S_Rib_FS_RS

    return S_Rib_LE_FS, S_Rib_FS_RS, S_Rib_total


if __name__ == "__main__":
    # Test the module
    print("=== Geometry Module Test ===\n")

    # Example planform (small UAV)
    planform = PlanformParams.from_input(
        b=2.0,              # 2m wingspan
        AR=8.0,             # aspect ratio
        taper_ratio=0.5,    # λ = 0.5
        t_c=0.12,           # 12% thickness
        S_ref=0.5,          # 0.5 m²
        C_r_mm=300,         # 300mm root chord
        c_MGC=0.25,         # 250mm MGC
        Y_bar_mm=400        # 400mm
    )

    print(f"Planform:")
    print(f"  Wingspan: {planform.b} m")
    print(f"  Half-span: {planform.L_span} m")
    print(f"  Root chord: {planform.C_r*1000:.1f} mm")
    print(f"  Tip chord: {planform.c_tip*1000:.1f} mm")
    print(f"  t/c: {planform.t_c}")

    # Spar positions
    spar_pos = SparPosition(X_FS_percent=25, X_RS_percent=65)
    print(f"\nSpar positions: FS={spar_pos.X_FS_percent}%, RS={spar_pos.X_RS_percent}%")
    print(f"  Valid: {spar_pos.validate()}")

    # Sweep angles
    sweep_FS = compute_spar_sweep_angle(spar_pos.X_FS_percent, planform)
    sweep_RS = compute_spar_sweep_angle(spar_pos.X_RS_percent, planform)
    print(f"  Sweep FS: {sweep_FS:.2f}°")
    print(f"  Sweep RS: {sweep_RS:.2f}°")

    # Rib stations
    N_Rib = 5
    stations = compute_all_stations(planform, spar_pos, N_Rib)
    print(f"\nRib stations (N_Rib={N_Rib}):")
    print(f"  {'y[mm]':>8} {'chord[mm]':>10} {'x_FS[mm]':>10} {'x_RS[mm]':>10} {'h_box[mm]':>10} {'A_m[mm²]':>12}")
    for s in stations:
        print(f"  {s.y*1000:8.1f} {s.chord*1000:10.1f} {s.x_FS*1000:10.1f} {s.x_RS*1000:10.1f} {s.h_box*1000:10.1f} {s.A_m*1e6:12.1f}")

    # Skin arc lengths
    arcs = compute_skin_arc_lengths_root(planform, spar_pos)
    print(f"\nSkin arc lengths (root):")
    print(f"  L_LE_FS: {arcs.L_LE_FS*1000:.1f} mm")
    print(f"  L_LE_RS: {arcs.L_LE_RS*1000:.1f} mm")
    print(f"  L_FS_RS: {arcs.L_FS_RS*1000:.1f} mm")

    # Skin area
    S_skin = compute_skin_area_half_wing(planform)
    print(f"\nHalf-wing skin area: {S_skin:.4f} m² = {S_skin*1e6:.0f} mm²")

    # Rib areas
    S_LE_FS, S_FS_RS, S_total = compute_rib_areas_root(planform, spar_pos)
    print(f"\nRib areas (root):")
    print(f"  S_Rib_LE_FS: {S_LE_FS*1e6:.1f} mm²")
    print(f"  S_Rib_FS_RS: {S_FS_RS*1e6:.1f} mm²")
    print(f"  S_Rib_total: {S_total*1e6:.1f} mm²")

    # Spar geometry test
    print("\n=== Spar Geometry Test ===")
    spar = SparGeometry.from_mm(d_outer_mm=20, t_wall_mm=2)
    print(f"  d_outer: {spar.d_outer*1000:.1f} mm")
    print(f"  d_inner: {spar.d_inner*1000:.1f} mm")
    print(f"  t_wall: {spar.t_wall*1000:.1f} mm")
    print(f"  Area: {spar.area*1e6:.2f} mm²")
    print(f"  I: {spar.I*1e12:.2f} mm⁴")
    print(f"  Valid: {spar.validate()}")

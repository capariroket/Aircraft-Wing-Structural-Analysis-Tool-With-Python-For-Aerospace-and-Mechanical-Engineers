"""
buckling.py - Panel and Rib Web Buckling Analysis Module

Implements:
- Skin panel shear buckling (k_s = 5.34, simply supported long plate)
- Skin panel compression buckling (k_c = 4.0, simply supported uniaxial)
- Combined shear-compression interaction: (tau/tau_cr)^2 + (sigma/sigma_cr) <= 1.0
- Rib web shear buckling (k_s = 5.34)
- Maximum spacing calculations for each mode

All internal calculations use SI units (Pa, m).
"""

import math
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# BUCKLING CRITICAL STRESSES
# =============================================================================

def skin_shear_buckling_critical(E: float, nu: float, t: float, s: float,
                                  k_s: float = 5.34) -> float:
    """
    Critical shear buckling stress for skin panel.

    tau_cr = k_s * pi^2 * E / (12*(1-nu^2)) * (t/s)^2

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        t: Skin thickness [m]
        s: Panel length / rib spacing [m]
        k_s: Shear buckling coefficient [-] (5.34 = SS, long plate)

    Returns:
        Critical shear buckling stress [Pa]
    """
    if s <= 0:
        return float('inf')
    return k_s * math.pi**2 * E / (12 * (1 - nu**2)) * (t / s)**2


def skin_compression_buckling_critical(E: float, nu: float, t: float, s: float,
                                        k_c: float = 4.0) -> float:
    """
    Critical compression buckling stress for skin panel.

    sigma_cr = k_c * pi^2 * E / (12*(1-nu^2)) * (t/s)^2

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        t: Skin thickness [m]
        s: Panel length / rib spacing [m]
        k_c: Compression buckling coefficient [-] (4.0 = SS, uniaxial)

    Returns:
        Critical compression buckling stress [Pa]
    """
    if s <= 0:
        return float('inf')
    return k_c * math.pi**2 * E / (12 * (1 - nu**2)) * (t / s)**2


def rib_web_shear_buckling_critical(E: float, nu: float, t_rib: float,
                                     h_box: float, k_s: float = 5.34) -> float:
    """
    Critical shear buckling stress for rib web.

    tau_cr_rib = k_s * pi^2 * E / (12*(1-nu^2)) * (t_rib/h_box)^2

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        t_rib: Rib web thickness [m]
        h_box: Box height (rib effective height) [m]
        k_s: Shear buckling coefficient [-] (5.34 = SS)

    Returns:
        Critical shear buckling stress [Pa]
    """
    if h_box <= 0:
        return float('inf')
    return k_s * math.pi**2 * E / (12 * (1 - nu**2)) * (t_rib / h_box)**2


# =============================================================================
# INTERACTION AND MARGINS
# =============================================================================

def combined_interaction_ratio(tau: float, tau_cr: float,
                                sigma: float, sigma_cr: float) -> float:
    """
    Combined shear-compression buckling interaction ratio.

    R = (tau/tau_cr)^2 + (sigma/sigma_cr)

    R < 1.0 means panel is safe under combined loading.

    Args:
        tau: Applied shear stress [Pa]
        tau_cr: Critical shear buckling stress [Pa]
        sigma: Applied compression stress [Pa]
        sigma_cr: Critical compression buckling stress [Pa]

    Returns:
        Interaction ratio R [-] (< 1.0 = pass)
    """
    R = 0.0
    if tau_cr > 0 and abs(tau) > 0:
        R += (tau / tau_cr)**2
    if sigma_cr > 0 and abs(sigma) > 0:
        R += abs(sigma) / sigma_cr
    return R


# =============================================================================
# MAXIMUM SPACING CALCULATIONS
# =============================================================================

def compute_s_max_shear(E: float, nu: float, t: float, tau_panel: float,
                         k_s: float = 5.34) -> float:
    """
    Maximum panel spacing to prevent shear buckling.

    s_max = t * sqrt(k_s * pi^2 * E / (12*(1-nu^2) * tau_panel))

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        t: Skin thickness [m]
        tau_panel: Applied shear stress [Pa]
        k_s: Shear buckling coefficient [-]

    Returns:
        Maximum spacing [m]
    """
    if abs(tau_panel) <= 0:
        return float('inf')
    return t * math.sqrt(k_s * math.pi**2 * E / (12 * (1 - nu**2) * abs(tau_panel)))


def compute_s_max_compression(E: float, nu: float, t: float, sigma_comp: float,
                                k_c: float = 4.0) -> float:
    """
    Maximum panel spacing to prevent compression buckling.

    s_max = t * sqrt(k_c * pi^2 * E / (12*(1-nu^2) * sigma_comp))

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        t: Skin thickness [m]
        sigma_comp: Applied compression stress [Pa]
        k_c: Compression buckling coefficient [-]

    Returns:
        Maximum spacing [m]
    """
    if abs(sigma_comp) <= 0:
        return float('inf')
    return t * math.sqrt(k_c * math.pi**2 * E / (12 * (1 - nu**2) * abs(sigma_comp)))


# =============================================================================
# COMPLETE BAY BUCKLING CHECK
# =============================================================================

@dataclass
class BayBucklingResult:
    """Complete buckling analysis result for a single panel bay."""
    bay_id: int
    y_start: float          # Bay start position [m]
    y_end: float            # Bay end position [m]
    spacing: float          # Bay spacing s_i [m]

    # Applied stresses
    tau_panel: float        # Panel shear stress [Pa]
    sigma_comp: float       # Skin compression stress [Pa]

    # Shear buckling
    tau_cr: float           # Critical shear buckling stress [Pa]
    shear_margin: float     # tau_cr / tau_panel [-]

    # Compression buckling
    sigma_cr: float         # Critical compression buckling stress [Pa]
    comp_margin: float      # sigma_cr / sigma_comp [-]

    # Combined interaction
    R_combined: float       # (tau/tau_cr)^2 + (sigma/sigma_cr) [-]

    # Allowable checks
    tau_allow: float        # Allowable shear stress [Pa]
    sigma_allow: float      # Allowable compression stress [Pa]

    # Pass/fail flags
    pass_shear_buckling: bool
    pass_comp_buckling: bool
    pass_combined: bool
    pass_shear_allowable: bool
    pass_comp_allowable: bool

    # Rib web buckling at bay endpoints
    tau_cr_rib_start: float = 0.0
    tau_cr_rib_end: float = 0.0
    rib_shear_start: float = 0.0   # V / (h_box * t_rib) at start
    rib_shear_end: float = 0.0     # V / (h_box * t_rib) at end
    pass_rib_start: bool = True
    pass_rib_end: bool = True

    # Required spacing
    s_max_shear: float = float('inf')
    s_max_comp: float = float('inf')

    @property
    def pass_all(self) -> bool:
        """True if all checks pass."""
        return (self.pass_shear_buckling and self.pass_comp_buckling and
                self.pass_combined and self.pass_shear_allowable and
                self.pass_comp_allowable and
                self.pass_rib_start and self.pass_rib_end)

    @property
    def fail_reasons(self) -> list:
        """List of failed criteria."""
        reasons = []
        if not self.pass_shear_buckling:
            reasons.append("SKIN_SHEAR_BUCKLING")
        if not self.pass_comp_buckling:
            reasons.append("SKIN_COMP_BUCKLING")
        if not self.pass_combined:
            reasons.append("COMBINED_INTERACTION")
        if not self.pass_shear_allowable:
            reasons.append("SKIN_SHEAR_ALLOWABLE")
        if not self.pass_comp_allowable:
            reasons.append("SKIN_COMP_ALLOWABLE")
        if not self.pass_rib_start:
            reasons.append("RIB_BUCKLING_START")
        if not self.pass_rib_end:
            reasons.append("RIB_BUCKLING_END")
        return reasons


def check_bay_buckling(bay_id: int, y_start: float, y_end: float,
                        tau_panel: float, sigma_comp: float,
                        tau_allow: float, sigma_allow: float,
                        E_skin: float, nu_skin: float, t_skin: float,
                        k_s: float = 5.34, k_c: float = 4.0,
                        # Rib web parameters (optional)
                        V_start: float = 0.0, V_end: float = 0.0,
                        h_box_start: float = 0.0, h_box_end: float = 0.0,
                        t_rib: float = 0.0, E_rib: float = 0.0,
                        nu_rib: float = 0.0) -> BayBucklingResult:
    """
    Complete buckling check for a single panel bay.

    Checks:
    1. Skin shear buckling: tau_panel <= tau_cr
    2. Skin compression buckling: sigma_comp <= sigma_cr
    3. Combined interaction: R = (tau/tau_cr)^2 + (sigma/sigma_cr) < 1.0
    4. Allowable stress limits
    5. Rib web shear buckling at bay endpoints

    Args:
        bay_id: Bay identifier
        y_start, y_end: Bay extent [m]
        tau_panel: Panel shear stress [Pa]
        sigma_comp: Skin compression stress [Pa]
        tau_allow: Allowable shear stress [Pa]
        sigma_allow: Allowable compression stress [Pa]
        E_skin, nu_skin, t_skin: Skin material/geometry
        k_s, k_c: Buckling coefficients
        V_start, V_end: Shear force at endpoints [N]
        h_box_start, h_box_end: Box height at endpoints [m]
        t_rib, E_rib, nu_rib: Rib material/geometry

    Returns:
        BayBucklingResult with all checks
    """
    spacing = y_end - y_start

    # Skin buckling
    tau_cr = skin_shear_buckling_critical(E_skin, nu_skin, t_skin, spacing, k_s)
    sigma_cr = skin_compression_buckling_critical(E_skin, nu_skin, t_skin, spacing, k_c)
    R = combined_interaction_ratio(tau_panel, tau_cr, sigma_comp, sigma_cr)

    shear_margin = tau_cr / abs(tau_panel) if abs(tau_panel) > 0 else float('inf')
    comp_margin = sigma_cr / abs(sigma_comp) if abs(sigma_comp) > 0 else float('inf')

    # Required spacing
    s_max_shear = compute_s_max_shear(E_skin, nu_skin, t_skin, tau_panel, k_s)
    s_max_comp = compute_s_max_compression(E_skin, nu_skin, t_skin, sigma_comp, k_c)

    # Rib web shear buckling
    tau_cr_rib_s = 0.0
    tau_cr_rib_e = 0.0
    rib_shear_s = 0.0
    rib_shear_e = 0.0
    pass_rib_s = True
    pass_rib_e = True

    if t_rib > 0 and E_rib > 0:
        if h_box_start > 0:
            tau_cr_rib_s = rib_web_shear_buckling_critical(E_rib, nu_rib, t_rib, h_box_start)
            rib_shear_s = abs(V_start) / (h_box_start * t_rib) if h_box_start > 0 else 0.0
            pass_rib_s = rib_shear_s <= tau_cr_rib_s
        if h_box_end > 0:
            tau_cr_rib_e = rib_web_shear_buckling_critical(E_rib, nu_rib, t_rib, h_box_end)
            rib_shear_e = abs(V_end) / (h_box_end * t_rib) if h_box_end > 0 else 0.0
            pass_rib_e = rib_shear_e <= tau_cr_rib_e

    return BayBucklingResult(
        bay_id=bay_id,
        y_start=y_start,
        y_end=y_end,
        spacing=spacing,
        tau_panel=abs(tau_panel),
        sigma_comp=abs(sigma_comp),
        tau_cr=tau_cr,
        shear_margin=shear_margin,
        sigma_cr=sigma_cr,
        comp_margin=comp_margin,
        R_combined=R,
        tau_allow=tau_allow,
        sigma_allow=sigma_allow,
        pass_shear_buckling=(abs(tau_panel) <= tau_cr),
        pass_comp_buckling=(abs(sigma_comp) <= sigma_cr),
        pass_combined=(R < 1.0),
        pass_shear_allowable=(abs(tau_panel) <= tau_allow),
        pass_comp_allowable=(abs(sigma_comp) <= sigma_allow),
        tau_cr_rib_start=tau_cr_rib_s,
        tau_cr_rib_end=tau_cr_rib_e,
        rib_shear_start=rib_shear_s,
        rib_shear_end=rib_shear_e,
        pass_rib_start=pass_rib_s,
        pass_rib_end=pass_rib_e,
        s_max_shear=s_max_shear,
        s_max_comp=s_max_comp,
    )


if __name__ == "__main__":
    print("=== Buckling Module Test ===\n")

    # PLA skin properties
    E_pla = 3.5e9       # 3.5 GPa
    nu_pla = 0.36
    t_skin = 0.625e-3   # 0.625 mm

    # Test: shear buckling for different spacings
    print("Shear buckling (PLA, t=0.625mm):")
    for s_mm in [50, 100, 200, 300, 500]:
        s = s_mm / 1000
        tau_cr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, s)
        print(f"  s={s_mm:4d}mm: tau_cr = {tau_cr/1e6:.3f} MPa")

    # Test: compression buckling
    print("\nCompression buckling (PLA, t=0.625mm):")
    for s_mm in [50, 100, 200, 300, 500]:
        s = s_mm / 1000
        sigma_cr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, s)
        print(f"  s={s_mm:4d}mm: sigma_cr = {sigma_cr/1e6:.3f} MPa")

    # Test: combined interaction
    print("\nCombined interaction test:")
    tau_cr = skin_shear_buckling_critical(E_pla, nu_pla, t_skin, 0.2)
    sigma_cr = skin_compression_buckling_critical(E_pla, nu_pla, t_skin, 0.2)
    tau_app = tau_cr * 0.5
    sigma_app = sigma_cr * 0.3
    R = combined_interaction_ratio(tau_app, tau_cr, sigma_app, sigma_cr)
    print(f"  tau/tau_cr=0.5, sigma/sigma_cr=0.3")
    print(f"  R = {R:.4f} ({'PASS' if R < 1.0 else 'FAIL'})")

    # Test: rib web
    print("\nRib web shear buckling (PLA, t_rib=3mm):")
    t_rib = 3e-3
    for h_mm in [20, 40, 60, 80]:
        h = h_mm / 1000
        tau_cr_rib = rib_web_shear_buckling_critical(E_pla, nu_pla, t_rib, h)
        print(f"  h_box={h_mm:3d}mm: tau_cr_rib = {tau_cr_rib/1e6:.2f} MPa")

    # Test: max spacing
    print("\nMax spacing (PLA, t=0.625mm, tau=0.1MPa):")
    s_max_sh = compute_s_max_shear(E_pla, nu_pla, t_skin, 0.1e6)
    s_max_co = compute_s_max_compression(E_pla, nu_pla, t_skin, 0.1e6)
    print(f"  s_max_shear = {s_max_sh*1000:.1f} mm")
    print(f"  s_max_comp  = {s_max_co*1000:.1f} mm")

"""
materials.py - Material Database and Selection Module

Handles material properties, selection, and allowable stress calculations.
All internal calculations use SI units (Pa, kg/m³).
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Material:
    """Material properties container."""
    name: str
    E: float           # Young's modulus [Pa]
    nu: float          # Poisson's ratio [-]
    density: float     # Density [kg/m³]
    sigma_u: float     # Ultimate tensile strength [Pa]
    tau_u: float       # Ultimate shear strength [Pa]
    G: Optional[float] = None  # Shear modulus [Pa], calculated if not provided

    def __post_init__(self):
        """Calculate shear modulus if not provided."""
        if self.G is None:
            self.G = self.E / (2 * (1 + self.nu))

    def get_allowables(self, SF: float) -> Dict[str, float]:
        """
        Calculate allowable stresses with safety factor.

        Args:
            SF: Safety factor [-]

        Returns:
            Dictionary with sigma_allow and tau_allow [Pa]
        """
        return {
            'sigma_allow': self.sigma_u / SF,
            'tau_allow': self.tau_u / SF
        }


# Default material database
DEFAULT_MATERIALS: Dict[str, Material] = {
    'AL7075-T6': Material(
        name='Aluminum 7075-T6',
        E=71.7e9,           # 71.7 GPa
        nu=0.33,
        density=2810,       # kg/m³
        sigma_u=572e6,      # 572 MPa
        tau_u=331e6         # 331 MPa
    ),
    'AL2024-T3': Material(
        name='Aluminum 2024-T3',
        E=73.1e9,           # 73.1 GPa
        nu=0.33,
        density=2780,       # kg/m³
        sigma_u=483e6,      # 483 MPa
        tau_u=283e6         # 283 MPa
    ),
    'CFRP_UD': Material(
        name='Carbon Fiber UD (longitudinal)',
        E=145e9,            # 145 GPa
        nu=0.3,
        density=1500,       # kg/m³
        sigma_u=2080e6,     # 2080 MPa
        tau_u=500e6          # 500 MPa (in-plane shear, CFRP tube)
    ),
    'GFRP': Material(
        name='Glass Fiber Reinforced Polymer',
        E=40e9,             # 40 GPa
        nu=0.28,
        density=1900,       # kg/m³
        sigma_u=600e6,      # 600 MPa
        tau_u=60e6          # 60 MPa
    ),
    'PLA': Material(
        name='PLA (3D Print)',
        E=3.5e9,            # 3.5 GPa
        nu=0.36,
        density=1250,       # kg/m³
        sigma_u=50e6,       # 50 MPa
        tau_u=30e6          # 30 MPa
    ),
    'STEEL_4130': Material(
        name='Steel 4130 (normalized)',
        E=205e9,            # 205 GPa
        nu=0.29,
        density=7850,       # kg/m³
        sigma_u=670e6,      # 670 MPa
        tau_u=400e6         # 400 MPa
    ),
}


class MaterialDatabase:
    """Material database manager with selection capabilities."""

    def __init__(self, custom_materials: Optional[Dict[str, Material]] = None):
        """
        Initialize database with default and optional custom materials.

        Args:
            custom_materials: Optional dictionary of additional materials
        """
        self.materials = DEFAULT_MATERIALS.copy()
        if custom_materials:
            self.materials.update(custom_materials)

    def add_material(self, key: str, material: Material) -> None:
        """Add a material to the database."""
        self.materials[key] = material

    def get_material(self, key: str) -> Material:
        """
        Get material by key.

        Args:
            key: Material identifier

        Returns:
            Material object

        Raises:
            KeyError: If material not found
        """
        if key not in self.materials:
            available = ', '.join(self.materials.keys())
            raise KeyError(f"Material '{key}' not found. Available: {available}")
        return self.materials[key]

    def list_materials(self) -> Dict[str, str]:
        """List all available materials with their full names."""
        return {k: v.name for k, v in self.materials.items()}

    def add_material_from_dict(self, key: str, props: Dict) -> None:
        """
        Add material from dictionary (e.g., JSON input).

        Args:
            key: Material identifier
            props: Dictionary with material properties
                   Required: name, E, nu, density, sigma_u, tau_u
                   Optional: G
        """
        material = Material(
            name=props['name'],
            E=props['E'],
            nu=props['nu'],
            density=props['density'],
            sigma_u=props['sigma_u'],
            tau_u=props['tau_u'],
            G=props.get('G')
        )
        self.materials[key] = material


@dataclass
class MaterialSelection:
    """Selected materials for wing components."""
    spar: Material
    skin: Material
    rib: Material

    @classmethod
    def from_database(cls, db: MaterialDatabase,
                      spar_key: str, skin_key: str, rib_key: str) -> 'MaterialSelection':
        """
        Create selection from database keys.

        Args:
            db: MaterialDatabase instance
            spar_key: Material key for spars
            skin_key: Material key for skin
            rib_key: Material key for ribs

        Returns:
            MaterialSelection instance
        """
        return cls(
            spar=db.get_material(spar_key),
            skin=db.get_material(skin_key),
            rib=db.get_material(rib_key)
        )


# Unit conversion helpers (user interface uses mm and MPa)
def mpa_to_pa(value_mpa: float) -> float:
    """Convert MPa to Pa."""
    return value_mpa * 1e6


def pa_to_mpa(value_pa: float) -> float:
    """Convert Pa to MPa."""
    return value_pa / 1e6


def mm_to_m(value_mm: float) -> float:
    """Convert mm to m."""
    return value_mm / 1000


def m_to_mm(value_m: float) -> float:
    """Convert m to mm."""
    return value_m * 1000


if __name__ == "__main__":
    # Test the module
    db = MaterialDatabase()
    print("Available materials:")
    for key, name in db.list_materials().items():
        print(f"  {key}: {name}")

    print("\nAL7075-T6 properties:")
    al = db.get_material('AL7075-T6')
    print(f"  E = {al.E/1e9:.1f} GPa")
    print(f"  G = {al.G/1e9:.1f} GPa")
    print(f"  density = {al.density} kg/m³")
    print(f"  sigma_u = {al.sigma_u/1e6:.0f} MPa")

    print("\nAllowables with SF=1.5:")
    allowables = al.get_allowables(SF=1.5)
    print(f"  sigma_allow = {allowables['sigma_allow']/1e6:.1f} MPa")
    print(f"  tau_allow = {allowables['tau_allow']/1e6:.1f} MPa")

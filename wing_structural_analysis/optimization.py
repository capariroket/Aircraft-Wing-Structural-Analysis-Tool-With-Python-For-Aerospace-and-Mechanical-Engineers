"""
optimization.py - Grid Search Optimization Module

Performs grid search over design parameters to find minimum-weight
configuration that satisfies all structural constraints.

Integrates:
- Box beam model (transformed section) for bending stiffness
- Buckling checks (skin shear, skin compression, combined interaction)
- Adaptive rib insertion (bisection until all bays pass buckling)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from itertools import product
import time
import multiprocessing as mp
from scipy.interpolate import interp1d

from materials import Material, MaterialSelection, MaterialDatabase
from geometry import (PlanformParams, SparPosition, SparGeometry,
                      compute_all_stations, compute_spar_sweep_angle,
                      compute_skin_arc_lengths_root, compute_skin_area_half_wing,
                      chord_at_station as chord_at_station_val)
from loads import (FlightCondition, AeroCenter, analyze_loads,
                   LoadDistributionType, PitchMomentDistributionType,
                   compute_root_reactions)
from torsion import analyze_all_stations as analyze_torsion_stations, find_critical_station
from spars import (SparProperties, LoadSharing, analyze_spars_all_stations,
                   find_critical_spar_station, spar_mass, tip_deflection_uniform_load,
                   tip_deflection_from_moment, critical_area_bending,
                   BoxBeamSection, compute_box_beam_section,
                   analyze_box_beam_stress, find_critical_box_beam_station)
from buckling import (check_bay_buckling, skin_shear_buckling_critical,
                       skin_compression_buckling_critical, combined_interaction_ratio,
                       rib_web_shear_buckling_critical)
from ribs import (RibProperties, generate_rib_geometries, compute_total_rib_mass,
                  compute_rib_spacing, AdaptiveRibConfig, adaptive_rib_insertion,
                  compute_bay_results, compute_minimum_rib_count,
                  fix_rib_web_buckling)


@dataclass
class DesignRange:
    """Range definition for a design variable."""
    min_val: float
    max_val: float
    step: float

    def values(self) -> np.ndarray:
        """Generate array of values in range."""
        return np.arange(self.min_val, self.max_val + self.step / 2, self.step)

    @property
    def n_values(self) -> int:
        """Number of discrete values."""
        return len(self.values())


@dataclass
class DesignSpace:
    """Complete design space definition."""
    # Rib parameters
    N_Rib: DesignRange
    t_rib_mm: DesignRange

    # Spar positions (% chord)
    X_FS_percent: DesignRange
    X_RS_percent: DesignRange

    # Front spar geometry (mm)
    d_FS_outer_mm: DesignRange
    t_FS_mm: DesignRange

    # Rear spar geometry (mm)
    d_RS_outer_mm: DesignRange
    t_RS_mm: DesignRange

    @property
    def total_combinations(self) -> int:
        """Total number of combinations."""
        return (self.N_Rib.n_values *
                self.t_rib_mm.n_values *
                self.X_FS_percent.n_values *
                self.X_RS_percent.n_values *
                self.d_FS_outer_mm.n_values *
                self.t_FS_mm.n_values *
                self.d_RS_outer_mm.n_values *
                self.t_RS_mm.n_values)


@dataclass
class ConfigCandidate:
    """A candidate configuration."""
    N_Rib: int
    t_rib_mm: float
    X_FS_percent: float
    X_RS_percent: float
    d_FS_outer_mm: float
    t_FS_mm: float
    d_RS_outer_mm: float
    t_RS_mm: float


@dataclass
class EvaluationResult:
    """Result of evaluating a configuration."""
    config: ConfigCandidate
    valid: bool
    accepted: bool
    rejection_reason: str = ""
    mass_total: float = 0.0
    mass_skin: float = 0.0
    mass_FS: float = 0.0
    mass_RS: float = 0.0
    mass_ribs: float = 0.0
    tau_skin_max: float = 0.0
    # Spar stresses (for interaction formula)
    sigma_b_FS_max: float = 0.0  # Bending stress FS [Pa]
    tau_FS_max: float = 0.0       # Shear stress FS [Pa]
    sigma_b_RS_max: float = 0.0  # Bending stress RS [Pa]
    tau_RS_max: float = 0.0       # Shear stress RS [Pa]
    sigma_vm_FS_max: float = 0.0  # Von Mises FS (for reference)
    sigma_vm_RS_max: float = 0.0  # Von Mises RS (for reference)
    # Areas
    A_Act_FS: float = 0.0  # Actual area FS [m^2]
    A_Act_RS: float = 0.0  # Actual area RS [m^2]
    A_Cri_FS: float = 0.0  # Critical area FS [m^2]
    A_Cri_RS: float = 0.0  # Critical area RS [m^2]
    delta_tip: float = 0.0
    theta_tip_deg: float = 0.0  # Tip twist angle [degrees]
    eta_FS: float = 0.0
    eta_RS: float = 0.0
    I_FS: float = 0.0
    I_RS: float = 0.0
    # Box beam model results
    I_box_root: float = 0.0      # Box beam I at root [m^4]
    sigma_skin_comp_max: float = 0.0  # Max skin compression [Pa]
    skin_fraction: float = 0.0   # Skin I fraction [-]
    # Buckling results
    tau_cr_min: float = 0.0      # Min shear buckling critical [Pa]
    sigma_cr_min: float = 0.0    # Min compression buckling critical [Pa]
    R_combined_max: float = 0.0  # Max combined interaction ratio [-]
    # Adaptive rib results
    N_Rib_final: int = 0         # Final rib count after adaptive insertion
    rib_feasible: bool = True    # Whether adaptive insertion found feasible solution
    # Phase-2 results
    t_skin_final_mm: float = 0.0  # Final skin thickness after Phase-2 [mm]
    phase2_adjustments: int = 0   # Number of spacing adjustments in Phase-2
    phase2_ribs_added: int = 0    # Number of ribs added in Phase-2
    final_y_ribs: Optional[np.ndarray] = None  # Final rib positions [m]
    # Rib web buckling results
    t_rib_per_station: Optional[np.ndarray] = None  # Per-station t_rib [m]
    rib_web_n_thickened: int = 0   # Number of ribs thickened for web buckling
    t_rib_max_mm: float = 0.0     # Max t_rib after web buckling fix [mm]


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    # Fixed parameters
    planform: PlanformParams
    flight: FlightCondition
    aero_center: AeroCenter
    materials: MaterialSelection
    t_skin_mm: float
    SF: float

    # Spar web heights for assembly check [mm]
    h_FS_mm: float = 0.0  # Front spar web height (will be calculated from t/c * C_r if 0)
    h_RS_mm: float = 0.0  # Rear spar web height (will be calculated from t/c * C_r if 0)

    # Tip twist constraint [degrees]
    theta_max_deg: float = 2.0  # Maximum allowable tip twist (default 2 deg)

    # Rib profile type
    rib_parabolic: bool = False  # True=parabolic, False=rectangular

    # Distribution types
    load_dist_type: LoadDistributionType = LoadDistributionType.ELLIPTIC
    pitch_dist_type: PitchMomentDistributionType = PitchMomentDistributionType.CHORD_WEIGHTED

    # Buckling coefficients
    k_s: float = 10.0           # Skin shear buckling coefficient
    k_c: float = 8.0            # Skin compression buckling coefficient
    s_min_mm: float = 20.0      # Minimum practical rib spacing [mm]
    buckling_margin: float = 0.95  # Safety margin for adaptive insertion
    buckling_mode: int = 1      # 1=shear only, 2=shear+compression


class GridSearchOptimizer:
    """Grid search optimizer for wing structural sizing."""

    def __init__(self, opt_config: OptimizationConfig, design_space: DesignSpace):
        self.opt_config = opt_config
        self.design_space = design_space
        self.results: List[EvaluationResult] = []
        self.accepted_results: List[EvaluationResult] = []
        self.rejection_counts: Dict[str, int] = {}
        # Pre-compute allowables once (materials and SF don't change between configs)
        self._allowables = self.compute_allowables()

    def validate_geometry(self, config: ConfigCandidate) -> Tuple[bool, str]:
        """Validate geometric constraints."""
        # Check spar positions
        if config.X_FS_percent >= config.X_RS_percent:
            return False, "X_FS >= X_RS"

        # Check spar wall thickness vs diameter
        if config.t_FS_mm >= config.d_FS_outer_mm / 2:
            return False, "t_FS >= d_FS/2"
        if config.t_RS_mm >= config.d_RS_outer_mm / 2:
            return False, "t_RS >= d_RS/2"

        # Check positive values
        if config.d_FS_outer_mm <= 0 or config.d_RS_outer_mm <= 0:
            return False, "d <= 0"
        if config.t_FS_mm <= 0 or config.t_RS_mm <= 0:
            return False, "t <= 0"
        if config.t_rib_mm <= 0:
            return False, "t_rib <= 0"

        # Check N_Rib
        if config.N_Rib < 2:
            return False, "N_Rib < 2"

        return True, ""

    def compute_allowables(self) -> Dict[str, float]:
        """Compute allowable stresses."""
        SF = self.opt_config.SF
        mats = self.opt_config.materials

        return {
            'sigma_allow_spar': mats.spar.sigma_u / SF,
            'tau_allow_spar': mats.spar.tau_u / SF,
            'sigma_allow_skin': mats.skin.sigma_u / SF,
            'tau_allow_skin': mats.skin.tau_u / SF,
            'sigma_allow_rib': mats.rib.sigma_u / SF,
            'tau_allow_rib': mats.rib.tau_u / SF,
        }

    def evaluate_config(self, config: ConfigCandidate) -> EvaluationResult:
        """Phase-1: Evaluate a configuration WITHOUT buckling checks.

        Fast acceptance checks only — no buckling, no adaptive rib insertion.
        Buckling is handled in Phase-2 for the Top-20 configurations.

        Phase 0: Geometry validation + assembly check
        Phase 1: Loads + torsion → skin stress & twist
        Phase 2: Box beam → spar interaction & area
        Mass computation (for ranking)
        """
        result = EvaluationResult(config=config, valid=False, accepted=False)

        # =====================================================================
        # PHASE 0: Geometry validation + assembly (NO analysis needed)
        # =====================================================================
        valid, reason = self.validate_geometry(config)
        if not valid:
            result.rejection_reason = reason
            return result

        result.valid = True
        planform = self.opt_config.planform

        # 8.6/8.7 Assembly check (only needs diameter + h_box_mm)
        h_FS_mm = self.opt_config.h_FS_mm
        if h_FS_mm <= 0:
            h_FS_mm = planform.t_c * planform.C_r * 1000
        h_RS_mm = self.opt_config.h_RS_mm
        if h_RS_mm <= 0:
            h_RS_mm = planform.t_c * planform.C_r * 1000

        if config.d_FS_outer_mm + 10 >= h_FS_mm:
            result.rejection_reason = "FS assembly failed"
            return result
        if config.d_RS_outer_mm + 10 >= h_RS_mm:
            result.rejection_reason = "RS assembly failed"
            return result

        # =====================================================================
        # PHASE 1: Geometry + loads + torsion → skin stress & twist checks
        # =====================================================================
        spar_pos = SparPosition(
            X_FS_percent=config.X_FS_percent,
            X_RS_percent=config.X_RS_percent
        )
        stations = compute_all_stations(planform, spar_pos, config.N_Rib)

        y = np.array([s.y for s in stations])
        chord = np.array([s.chord for s in stations])
        x_FS = np.array([s.x_FS for s in stations])
        x_RS = np.array([s.x_RS for s in stations])
        A_m = np.array([s.A_m for s in stations])
        h_box = np.array([s.h_box for s in stations])
        box_width = np.array([s.width for s in stations])

        loads = analyze_loads(
            y, chord, x_FS, x_RS,
            self.opt_config.flight,
            self.opt_config.aero_center,
            planform.L_span,
            self.opt_config.load_dist_type,
            self.opt_config.pitch_dist_type
        )

        t_skin = self.opt_config.t_skin_mm / 1000
        G_skin = self.opt_config.materials.skin.G

        torsion_results = analyze_torsion_stations(
            y, loads.T, A_m, t_skin, G_skin, box_width, h_box
        )

        _, crit_torsion = find_critical_station(torsion_results)
        result.tau_skin_max = abs(crit_torsion.tau_skin)

        twist_rates = np.array([r.twist_rate for r in torsion_results])
        theta_tip_rad = np.trapz(twist_rates, y)
        result.theta_tip_deg = np.degrees(abs(theta_tip_rad))

        # Early check: 8.1 Skin shear stress
        if result.tau_skin_max > self._allowables['tau_allow_skin']:
            result.rejection_reason = "Skin stress exceeded"
            return result

        # Early check: 8.8 Tip twist
        if result.theta_tip_deg > self.opt_config.theta_max_deg:
            result.rejection_reason = "Twist exceeded"
            return result

        # =====================================================================
        # PHASE 2: Spar properties + box beam → spar interaction & area checks
        # =====================================================================
        mat_spar = self.opt_config.materials.spar

        spar_FS = SparProperties.from_mm(
            d_outer_mm=config.d_FS_outer_mm,
            t_wall_mm=config.t_FS_mm,
            material=mat_spar
        )
        spar_RS = SparProperties.from_mm(
            d_outer_mm=config.d_RS_outer_mm,
            t_wall_mm=config.t_RS_mm,
            material=mat_spar
        )

        result.A_Act_FS = spar_FS.area
        result.A_Act_RS = spar_RS.area

        E_skin = self.opt_config.materials.skin.E
        box_sections = [
            compute_box_beam_section(spar_FS, spar_RS, h_box[i], box_width[i], t_skin, E_skin)
            for i in range(len(y))
        ]

        sharing = LoadSharing.from_positions(
            config.X_FS_percent, config.X_RS_percent,
            self.opt_config.aero_center.x_ac_percent
        )

        box_results = [
            analyze_box_beam_stress(loads.M[i], loads.T[i], box_sections[i],
                                    spar_FS, spar_RS, sharing.eta_FS, sharing.eta_RS, y[i])
            for i in range(len(y))
        ]

        # Find critical station
        max_vm_idx = 0
        max_vm = max(box_results[0].sigma_vm_FS, box_results[0].sigma_vm_RS)
        max_skin_comp = box_results[0].sigma_skin_comp
        for i, br in enumerate(box_results):
            vm = max(br.sigma_vm_FS, br.sigma_vm_RS)
            if vm > max_vm:
                max_vm = vm
                max_vm_idx = i
            if br.sigma_skin_comp > max_skin_comp:
                max_skin_comp = br.sigma_skin_comp

        crit_box = box_results[max_vm_idx]

        result.sigma_b_FS_max = crit_box.sigma_b_FS
        result.tau_FS_max = crit_box.tau_FS
        result.sigma_b_RS_max = crit_box.sigma_b_RS
        result.tau_RS_max = crit_box.tau_RS
        result.sigma_vm_FS_max = crit_box.sigma_vm_FS
        result.sigma_vm_RS_max = crit_box.sigma_vm_RS
        result.eta_FS = sharing.eta_FS
        result.eta_RS = sharing.eta_RS
        result.I_FS = spar_FS.I
        result.I_RS = spar_RS.I
        result.I_box_root = box_sections[0].I_total
        result.sigma_skin_comp_max = max_skin_comp
        result.skin_fraction = box_sections[0].skin_fraction

        # Critical areas
        n = self.opt_config.flight.n
        W_0 = self.opt_config.flight.W0
        Y_bar = planform.Y_bar
        sigma_max_spar = mat_spar.sigma_u
        c_FS = spar_FS.d_outer / 2
        c_RS = spar_RS.d_outer / 2

        result.A_Cri_FS = (n * W_0 * Y_bar * 0.5 * result.eta_FS) / (2 * sigma_max_spar * c_FS)
        result.A_Cri_RS = (n * W_0 * Y_bar * 0.5 * result.eta_RS) / (2 * sigma_max_spar * c_RS)

        # Early check: 8.2 FS interaction
        sigma_allow_spar = self._allowables['sigma_allow_spar']
        tau_allow_spar = self._allowables['tau_allow_spar']

        FS_interaction = (result.sigma_b_FS_max / sigma_allow_spar) + \
                         (result.tau_FS_max / tau_allow_spar) ** 2
        if FS_interaction >= 1.0:
            result.rejection_reason = "FS interaction exceeded"
            return result

        # Early check: 8.3 RS interaction
        RS_interaction = (result.sigma_b_RS_max / sigma_allow_spar) + \
                         (result.tau_RS_max / tau_allow_spar) ** 2
        if RS_interaction >= 1.0:
            result.rejection_reason = "RS interaction exceeded"
            return result

        # Early check: 8.4 FS area
        if result.A_Act_FS <= result.A_Cri_FS:
            result.rejection_reason = "FS area insufficient"
            return result

        # Early check: 8.5 RS area
        if result.A_Act_RS <= result.A_Cri_RS:
            result.rejection_reason = "RS area insufficient"
            return result

        # =====================================================================
        # MASS COMPUTATION (for Phase-1 ranking, no buckling)
        # =====================================================================
        I_total = box_sections[0].I_total
        E_equiv = box_sections[0].E_ref
        result.delta_tip = tip_deflection_from_moment(y, loads.M, E_equiv, I_total)

        S_skin = compute_skin_area_half_wing(planform)
        result.mass_skin = S_skin * t_skin * self.opt_config.materials.skin.density

        Lambda_FS = compute_spar_sweep_angle(
            config.X_FS_percent, planform,
            self.opt_config.aero_center.Lambda_ac_deg,
            self.opt_config.aero_center.x_ac_percent
        )
        Lambda_RS = compute_spar_sweep_angle(
            config.X_RS_percent, planform,
            self.opt_config.aero_center.Lambda_ac_deg,
            self.opt_config.aero_center.x_ac_percent
        )
        L_FS = planform.L_span / np.cos(np.radians(Lambda_FS))
        L_RS = planform.L_span / np.cos(np.radians(Lambda_RS))
        result.mass_FS = spar_mass(spar_FS, L_FS)
        result.mass_RS = spar_mass(spar_RS, L_RS)

        mat_rib = self.opt_config.materials.rib
        rib_props = RibProperties.from_mm(config.t_rib_mm, mat_rib)
        ribs = generate_rib_geometries(y, chord, h_box, x_FS, x_RS,
                                        parabolic=self.opt_config.rib_parabolic)
        rib_mass_result = compute_total_rib_mass(ribs, rib_props, planform.L_span)
        result.mass_ribs = rib_mass_result.total_rib_mass

        result.mass_total = result.mass_skin + result.mass_FS + result.mass_RS + result.mass_ribs
        result.N_Rib_final = config.N_Rib
        result.t_skin_final_mm = self.opt_config.t_skin_mm

        # Phase-1 accepted (buckling not checked yet)
        result.accepted = True
        return result

    def fix_buckling_phase2(self, result: EvaluationResult) -> EvaluationResult:
        """Phase-2: Fix buckling for a Phase-1 accepted configuration.

        Adjusts rib spacing using binary search. If buckling_mode=2
        (shear+compression) and N_Rib exceeds N_Rib_max, increases t_skin
        in 0.3mm steps until feasible or gives up.

        The configuration is never ELIMINATED — it is FIXED if possible,
        or marked as UNRESOLVABLE_BUCKLING.
        """
        config = result.config
        planform = self.opt_config.planform
        N_Rib_min = compute_minimum_rib_count(planform.AR, planform.S_ref, planform.c_MGC)
        N_Rib_max = 2 * N_Rib_min

        # Recompute geometry (needed for rib insertion)
        spar_pos = SparPosition(
            X_FS_percent=config.X_FS_percent,
            X_RS_percent=config.X_RS_percent
        )
        N_Rib_start = max(config.N_Rib, N_Rib_min)
        stations = compute_all_stations(planform, spar_pos, N_Rib_start)

        y = np.array([s.y for s in stations])
        chord = np.array([s.chord for s in stations])
        x_FS = np.array([s.x_FS for s in stations])
        x_RS = np.array([s.x_RS for s in stations])
        A_m = np.array([s.A_m for s in stations])
        h_box = np.array([s.h_box for s in stations])
        box_width = np.array([s.width for s in stations])

        loads = analyze_loads(
            y, chord, x_FS, x_RS,
            self.opt_config.flight,
            self.opt_config.aero_center,
            planform.L_span,
            self.opt_config.load_dist_type,
            self.opt_config.pitch_dist_type
        )

        mat_spar = self.opt_config.materials.spar
        spar_FS = SparProperties.from_mm(config.d_FS_outer_mm, config.t_FS_mm, mat_spar)
        spar_RS = SparProperties.from_mm(config.d_RS_outer_mm, config.t_RS_mm, mat_spar)
        E_skin = self.opt_config.materials.skin.E
        sharing = LoadSharing.from_positions(
            config.X_FS_percent, config.X_RS_percent,
            self.opt_config.aero_center.x_ac_percent)

        # Try different t_skin values (current, +0.3, +0.6, ...)
        t_skin_start_mm = self.opt_config.t_skin_mm
        t_skin_step_mm = 0.3
        t_skin_max_mm = 10.0  # Safety limit
        best_candidate = None
        best_mass = float('inf')

        t_skin_mm = t_skin_start_mm
        while t_skin_mm <= t_skin_max_mm:
            t_skin = t_skin_mm / 1000

            # Recompute torsion with this t_skin
            G_skin = self.opt_config.materials.skin.G
            torsion_results = analyze_torsion_stations(
                y, loads.T, A_m, t_skin, G_skin, box_width, h_box
            )

            # Recompute box beam with this t_skin
            box_sections = [
                compute_box_beam_section(spar_FS, spar_RS, h_box[i],
                                         box_width[i], t_skin, E_skin)
                for i in range(len(y))
            ]
            box_results = [
                analyze_box_beam_stress(loads.M[i], loads.T[i], box_sections[i],
                                        spar_FS, spar_RS, sharing.eta_FS,
                                        sharing.eta_RS, y[i])
                for i in range(len(y))
            ]

            # Build interpolation functions
            tau_skin_arr = np.array([abs(tr.tau_skin) for tr in torsion_results])
            sigma_comp_arr = np.array([br.sigma_skin_comp for br in box_results])
            tau_interp = interp1d(y, tau_skin_arr, kind='linear', fill_value='extrapolate')
            sigma_interp = interp1d(y, sigma_comp_arr, kind='linear', fill_value='extrapolate')
            V_interp = interp1d(y, loads.V, kind='linear', fill_value='extrapolate')
            h_interp = interp1d(y, h_box, kind='linear', fill_value='extrapolate')

            adapt_config = AdaptiveRibConfig(
                E_skin=E_skin,
                nu_skin=self.opt_config.materials.skin.nu,
                t_skin=t_skin,
                E_rib=self.opt_config.materials.rib.E,
                nu_rib=self.opt_config.materials.rib.nu,
                t_rib=config.t_rib_mm / 1000,
                tau_allow_skin=self._allowables['tau_allow_skin'],
                sigma_allow_skin=self._allowables['sigma_allow_skin'],
                k_s=self.opt_config.k_s,
                k_c=self.opt_config.k_c,
                s_min=self.opt_config.s_min_mm / 1000,
                margin=self.opt_config.buckling_margin,
                buckling_mode=self.opt_config.buckling_mode,
                N_Rib_max=N_Rib_max,
            )

            final_y_ribs, feasible, adapt_msg = adaptive_rib_insertion(
                y, adapt_config, tau_interp, sigma_interp, V_interp, h_interp
            )

            n_bays_final = len(final_y_ribs) - 1

            if feasible and n_bays_final <= N_Rib_max:
                # Compute mass for this (t_skin, N_Rib) combo
                S_skin = compute_skin_area_half_wing(planform)
                mass_skin = S_skin * t_skin * self.opt_config.materials.skin.density

                mat_rib = self.opt_config.materials.rib
                rib_props = RibProperties.from_mm(config.t_rib_mm, mat_rib)
                new_chord = np.array([
                    chord_at_station_val(yi, planform.C_r, planform.c_tip, planform.L_span)
                    for yi in final_y_ribs
                ])
                new_h_box = planform.t_c * new_chord
                new_x_FS = np.array([spar_pos.x_FS_at_station(c) for c in new_chord])
                new_x_RS = np.array([spar_pos.x_RS_at_station(c) for c in new_chord])
                new_ribs = generate_rib_geometries(final_y_ribs, new_chord, new_h_box,
                                                    new_x_FS, new_x_RS,
                                                    parabolic=self.opt_config.rib_parabolic)
                new_rib_mass = compute_total_rib_mass(new_ribs, rib_props, planform.L_span)

                total_mass = mass_skin + result.mass_FS + result.mass_RS + new_rib_mass.total_rib_mass

                if total_mass < best_mass:
                    best_mass = total_mass
                    best_candidate = {
                        't_skin_mm': t_skin_mm,
                        'N_Rib_final': n_bays_final,
                        'mass_skin': mass_skin,
                        'mass_ribs': new_rib_mass.total_rib_mass,
                        'mass_total': total_mass,
                        'final_y_ribs': final_y_ribs.copy(),
                        'adapt_msg': adapt_msg,
                        'tau_cr_min': 0.0,
                        'sigma_cr_min': 0.0,
                        'R_combined_max': 0.0,
                        'sigma_skin_comp_max': max(abs(sigma_interp(yr)) for yr in final_y_ribs),
                    }

                    # Compute buckling margins for reporting
                    for i in range(len(final_y_ribs) - 1):
                        y_s, y_e = final_y_ribs[i], final_y_ribs[i + 1]
                        sp = y_e - y_s
                        tau_p = max(abs(tau_interp(y_s)), abs(tau_interp(y_e)))
                        sig_p = max(abs(sigma_interp(y_s)), abs(sigma_interp(y_e)))
                        tc = skin_shear_buckling_critical(E_skin, adapt_config.nu_skin, t_skin, sp, adapt_config.k_s)
                        sc = skin_compression_buckling_critical(E_skin, adapt_config.nu_skin, t_skin, sp, adapt_config.k_c)
                        R = combined_interaction_ratio(tau_p, tc, sig_p, sc)
                        if best_candidate['tau_cr_min'] == 0 or tc < best_candidate['tau_cr_min']:
                            best_candidate['tau_cr_min'] = tc
                        if best_candidate['sigma_cr_min'] == 0 or sc < best_candidate['sigma_cr_min']:
                            best_candidate['sigma_cr_min'] = sc
                        if R > best_candidate['R_combined_max']:
                            best_candidate['R_combined_max'] = R

                # In shear-only mode, first feasible t_skin is optimal (thinner = lighter)
                if self.opt_config.buckling_mode == 1:
                    break

            # Only try thicker skin in shear+compression mode
            if self.opt_config.buckling_mode == 1:
                break  # Shear-only: if current t_skin fails, can't help by thickening
            t_skin_mm += t_skin_step_mm

        # Apply best candidate to result
        if best_candidate is not None:
            result.t_skin_final_mm = best_candidate['t_skin_mm']
            result.N_Rib_final = best_candidate['N_Rib_final']
            result.mass_skin = best_candidate['mass_skin']
            result.mass_ribs = best_candidate['mass_ribs']
            result.mass_total = best_candidate['mass_total']
            result.final_y_ribs = best_candidate['final_y_ribs']
            result.rib_feasible = True
            result.tau_cr_min = best_candidate['tau_cr_min']
            result.sigma_cr_min = best_candidate['sigma_cr_min']
            result.R_combined_max = best_candidate['R_combined_max']
            result.sigma_skin_comp_max = best_candidate['sigma_skin_comp_max']

            # --- Rib web buckling fix ---
            # Check each rib station and thicken t_rib where web buckles
            final_y = best_candidate['final_y_ribs']
            t_skin_best = best_candidate['t_skin_mm'] / 1000
            # Rebuild interpolators with the winning t_skin
            G_skin = self.opt_config.materials.skin.G
            tr_best = analyze_torsion_stations(
                y, loads.T, A_m, t_skin_best, G_skin, box_width, h_box)
            bs_best = [
                compute_box_beam_section(spar_FS, spar_RS, h_box[ii],
                                         box_width[ii], t_skin_best, E_skin)
                for ii in range(len(y))
            ]
            br_best = [
                analyze_box_beam_stress(loads.M[ii], loads.T[ii], bs_best[ii],
                                        spar_FS, spar_RS, sharing.eta_FS,
                                        sharing.eta_RS, y[ii])
                for ii in range(len(y))
            ]
            V_interp_best = interp1d(y, loads.V, kind='linear',
                                     fill_value='extrapolate')
            h_interp_best = interp1d(y, h_box, kind='linear',
                                     fill_value='extrapolate')

            rwb = fix_rib_web_buckling(
                final_y,
                V_at_y=V_interp_best,
                h_box_at_y=h_interp_best,
                E_rib=self.opt_config.materials.rib.E,
                nu_rib=self.opt_config.materials.rib.nu,
                t_rib_initial=config.t_rib_mm / 1000,
                t_rib_step=0.0005,   # 0.5 mm increments
                t_rib_max=0.010,     # 10 mm max
            )

            result.t_rib_per_station = rwb.t_rib_per_station
            result.rib_web_n_thickened = rwb.n_thickened
            result.t_rib_max_mm = rwb.t_rib_max_mm

            # Recompute rib mass using per-station t_rib
            if rwb.n_thickened > 0:
                mat_rib = self.opt_config.materials.rib
                new_chord = np.array([
                    chord_at_station_val(yi, planform.C_r, planform.c_tip,
                                         planform.L_span)
                    for yi in final_y
                ])
                new_h_box = planform.t_c * new_chord
                new_x_FS = np.array([spar_pos.x_FS_at_station(c)
                                     for c in new_chord])
                new_x_RS = np.array([spar_pos.x_RS_at_station(c)
                                     for c in new_chord])
                new_ribs = generate_rib_geometries(
                    final_y, new_chord, new_h_box, new_x_FS, new_x_RS,
                    parabolic=self.opt_config.rib_parabolic)

                # Weighted rib mass: each rib uses its own t_rib
                total_rib_mass = 0.0
                for ri in range(1, len(new_ribs)):  # skip root
                    t_ri = rwb.t_rib_per_station[ri]
                    total_rib_mass += new_ribs[ri].S_total * t_ri * mat_rib.density

                result.mass_ribs = total_rib_mass
                result.mass_total = (result.mass_skin + result.mass_FS +
                                     result.mass_RS + result.mass_ribs)

            result.accepted = True
        else:
            result.rib_feasible = False
            result.rejection_reason = "UNRESOLVABLE_BUCKLING"
            result.accepted = False

        return result

    def _generate_config_tuples(self) -> List[tuple]:
        """Generate all config tuples for evaluation."""
        configs = []
        for N_Rib in self.design_space.N_Rib.values().astype(int):
            for t_rib in self.design_space.t_rib_mm.values():
                for X_FS in self.design_space.X_FS_percent.values():
                    for X_RS in self.design_space.X_RS_percent.values():
                        for d_FS in self.design_space.d_FS_outer_mm.values():
                            for t_FS in self.design_space.t_FS_mm.values():
                                for d_RS in self.design_space.d_RS_outer_mm.values():
                                    for t_RS in self.design_space.t_RS_mm.values():
                                        configs.append((int(N_Rib), float(t_rib),
                                                        float(X_FS), float(X_RS),
                                                        float(d_FS), float(t_FS),
                                                        float(d_RS), float(t_RS)))
        return configs

    def _print_progress(self, count: int, total: int, start_time: float,
                        progress_interval: int):
        """Print progress with rejection diagnostics."""
        elapsed = time.time() - start_time
        rate = count / elapsed if elapsed > 0 else 0
        remaining = (total - count) / rate if rate > 0 else 0
        print(f"  Progress: {count}/{total} "
              f"({count/total*100:.1f}%) "
              f"- {len(self.accepted_results)} accepted "
              f"- ETA: {remaining:.0f}s")
        # Show top rejection reasons when 0 accepted (helps debug)
        if len(self.accepted_results) == 0 and self.rejection_counts:
            top = sorted(self.rejection_counts.items(), key=lambda x: -x[1])[:3]
            reasons = ", ".join(f"{r}:{c}" for r, c in top)
            print(f"    Top rejections: {reasons}")

    def run(self, progress_interval: int = 1000,
            n_workers: int = 1,
            top_n: int = 20) -> Optional[EvaluationResult]:
        """
        Run two-phase grid search optimization.

        Phase-1: Fast acceptance (no buckling) → find Top-20 by mass
        Phase-2: Fix buckling for each Top-20 → adjust ribs / thicken skin

        Args:
            progress_interval: Print progress every N iterations
            n_workers: Number of parallel workers (1 = sequential)
            top_n: Number of top configs to pass to Phase-2 (default 20)

        Returns:
            Best (minimum mass) accepted configuration, or None if no valid solution
        """
        total = self.design_space.total_combinations
        buckling_mode_name = ("Shear + Compression" if self.opt_config.buckling_mode == 2
                              else "Shear only")
        print(f"\n{'='*50}")
        print(f"PHASE-1: Fast Acceptance (no buckling)")
        print(f"{'='*50}")
        print(f"Total combinations: {total}")
        print(f"Buckling mode: {buckling_mode_name}")
        if n_workers > 1:
            print(f"Parallel workers: {n_workers}")

        start_time = time.time()

        if n_workers > 1:
            self._run_parallel(n_workers, progress_interval, start_time)
        else:
            self._run_sequential(progress_interval, start_time)

        elapsed_p1 = time.time() - start_time
        print(f"\nPhase-1 complete in {elapsed_p1:.1f}s")
        print(f"  Total evaluated: {len(self.results)}")
        print(f"  Valid geometry: {sum(1 for r in self.results if r.valid)}")
        print(f"  Phase-1 accepted: {len(self.accepted_results)}")

        if self.rejection_counts:
            total_rejected = sum(self.rejection_counts.values())
            print(f"\n  Rejection breakdown ({total_rejected} total):")
            for reason, count in sorted(self.rejection_counts.items(),
                                        key=lambda x: -x[1]):
                pct = count / total_rejected * 100
                print(f"    {pct:5.1f}% | {reason}")

        if not self.accepted_results:
            print("\nNo configurations pass Phase-1 acceptance checks.")
            print("Try expanding design space or relaxing constraints.")
            return None

        # =====================================================================
        # PHASE-2: Buckling fix for Top-N
        # =====================================================================
        top_configs = sorted(self.accepted_results, key=lambda r: r.mass_total)[:top_n]
        n_top = len(top_configs)

        planform = self.opt_config.planform
        N_Rib_min = compute_minimum_rib_count(planform.AR, planform.S_ref, planform.c_MGC)
        N_Rib_max = 2 * N_Rib_min

        print(f"\n{'='*50}")
        print(f"PHASE-2: Buckling Fix for Top-{n_top}")
        print(f"{'='*50}")
        print(f"  N_Rib_min = {N_Rib_min}, N_Rib_max = {N_Rib_max}")
        print(f"  t_skin_start = {self.opt_config.t_skin_mm:.1f} mm")
        if self.opt_config.buckling_mode == 2:
            print(f"  t_skin step = 0.3 mm (will increase if needed)")

        start_p2 = time.time()
        phase2_results = []
        n_resolved = 0
        n_unresolvable = 0

        for rank, p1_result in enumerate(top_configs, 1):
            p2_result = self.fix_buckling_phase2(p1_result)
            phase2_results.append(p2_result)

            cfg = p1_result.config
            if p2_result.accepted:
                n_resolved += 1
                status = "OK"
                detail = (f"N_Rib={p2_result.N_Rib_final}, "
                          f"t_skin={p2_result.t_skin_final_mm:.1f}mm, "
                          f"mass={p2_result.mass_total*1000:.1f}g")
            else:
                n_unresolvable += 1
                status = "UNRESOLVABLE"
                detail = p2_result.rejection_reason

            print(f"  [{rank:2d}/{n_top}] d_FS={cfg.d_FS_outer_mm:.0f} "
                  f"d_RS={cfg.d_RS_outer_mm:.0f} "
                  f"t_FS={cfg.t_FS_mm:.1f} t_RS={cfg.t_RS_mm:.1f} "
                  f"→ {status}: {detail}")

        elapsed_p2 = time.time() - start_p2
        print(f"\nPhase-2 complete in {elapsed_p2:.1f}s")
        print(f"  Resolved: {n_resolved}/{n_top}")
        print(f"  Unresolvable: {n_unresolvable}/{n_top}")

        # Find best Phase-2 result
        feasible_p2 = [r for r in phase2_results if r.accepted]
        if not feasible_p2:
            print(f"\nNo buckling-feasible solution among Top-{n_top}.")
            print("All configurations have UNRESOLVABLE_BUCKLING.")
            if self.opt_config.buckling_mode == 2:
                print("TIP: Try 'Shear only' buckling mode, or use stronger skin material.")
            return None

        # Update accepted_results to Phase-2 results
        self.accepted_results = feasible_p2

        best = min(feasible_p2, key=lambda r: r.mass_total)
        total_elapsed = time.time() - start_time
        print(f"\nBest solution: mass = {best.mass_total*1000:.2f} g")
        print(f"  t_skin = {best.t_skin_final_mm:.1f} mm, "
              f"N_Rib = {best.N_Rib_final}")
        print(f"Total optimization time: {total_elapsed:.1f}s")
        return best

    def _run_sequential(self, progress_interval: int, start_time: float):
        """Sequential grid search."""
        configs = self._generate_config_tuples()
        total = len(configs)

        for count, ct in enumerate(configs, 1):
            if count % progress_interval == 0:
                self._print_progress(count, total, start_time, progress_interval)

            config = ConfigCandidate(*ct)
            result = self.evaluate_config(config)
            self.results.append(result)

            if result.accepted:
                self.accepted_results.append(result)
            else:
                reason = result.rejection_reason or "Unknown"
                self.rejection_counts[reason] = \
                    self.rejection_counts.get(reason, 0) + 1

    def _run_parallel(self, n_workers: int, progress_interval: int,
                      start_time: float):
        """Parallel grid search using multiprocessing."""
        configs = self._generate_config_tuples()
        total = len(configs)
        chunk_size = max(1, total // (n_workers * 20))

        print(f"  Generating {total} configs, chunk_size={chunk_size}...")

        with mp.Pool(n_workers,
                     initializer=_init_pool_worker,
                     initargs=(self.opt_config, self.design_space)) as pool:
            count = 0
            for result in pool.imap_unordered(_eval_pool_worker, configs,
                                              chunksize=chunk_size):
                count += 1
                self.results.append(result)

                if result.accepted:
                    self.accepted_results.append(result)
                else:
                    reason = result.rejection_reason or "Unknown"
                    self.rejection_counts[reason] = \
                        self.rejection_counts.get(reason, 0) + 1

                if count % progress_interval == 0:
                    self._print_progress(count, total, start_time,
                                         progress_interval)

    def get_top_solutions(self, n: int = 10) -> List[EvaluationResult]:
        """Get top N solutions by mass."""
        sorted_results = sorted(self.accepted_results, key=lambda r: r.mass_total)
        return sorted_results[:n]

    def get_rejection_summary(self) -> Dict[str, int]:
        """Get summary of rejection reasons."""
        return self.rejection_counts.copy()


# =============================================================================
# MULTIPROCESSING WORKER FUNCTIONS (module-level for pickling)
# =============================================================================

_pool_optimizer = None

def _init_pool_worker(opt_config, design_space):
    """Initialize a GridSearchOptimizer in each worker process."""
    global _pool_optimizer
    _pool_optimizer = GridSearchOptimizer(opt_config, design_space)

def _eval_pool_worker(config_tuple):
    """Evaluate a single config tuple in a worker process."""
    config = ConfigCandidate(*config_tuple)
    return _pool_optimizer.evaluate_config(config)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_optimization(planform: PlanformParams,
                     flight: FlightCondition,
                     aero_center: AeroCenter,
                     materials: MaterialSelection,
                     design_space: DesignSpace,
                     t_skin_mm: float = 1.0,
                     SF: float = 1.5,
                     h_FS_mm: float = 0.0,
                     h_RS_mm: float = 0.0,
                     theta_max_deg: float = 2.0,
                     rib_parabolic: bool = False,
                     load_dist: str = "elliptic",
                     pitch_dist: str = "chord_weighted",
                     n_workers: int = 1,
                     s_min_mm: float = 20.0,
                     buckling_mode: int = 1) -> Tuple[Optional[EvaluationResult],
                                                       GridSearchOptimizer]:
    """
    Convenience function to run two-phase optimization.

    Phase-1: Fast grid search (no buckling) → Top-20
    Phase-2: Buckling fix for Top-20 (rib adjustment + optional t_skin increase)

    Args:
        h_FS_mm: Front spar web height [mm] (0 = auto-calculate from t/c * C_r)
        h_RS_mm: Rear spar web height [mm] (0 = auto-calculate from t/c * C_r)
        theta_max_deg: Maximum allowable tip twist [degrees] (default 2 deg)
        rib_parabolic: Use parabolic rib profile (True) or rectangular (False)
        n_workers: Number of parallel workers (1 = sequential, >1 = multiprocessing)
        buckling_mode: 1=shear only, 2=shear+compression

    Returns:
        Tuple of (best_result, optimizer)
    """
    load_dist_type = (LoadDistributionType.ELLIPTIC if load_dist == "elliptic"
                      else LoadDistributionType.UNIFORM)
    pitch_dist_type = (PitchMomentDistributionType.CHORD_WEIGHTED if pitch_dist == "chord_weighted"
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
    )

    optimizer = GridSearchOptimizer(opt_config, design_space)
    best = optimizer.run(n_workers=n_workers)

    return best, optimizer


if __name__ == "__main__":
    print("=== Optimization Module Test ===\n")

    # Setup planform
    planform = PlanformParams.from_input(
        b=2.0, AR=8.0, taper_ratio=0.5, t_c=0.12,
        S_ref=0.5, C_r_mm=300, c_MGC=0.25, Y_bar_mm=400
    )

    # Flight condition
    flight = FlightCondition(
        W0=50, n=2.0, V_c=25, rho=1.225,
        C_m=-0.05, S_ref=0.5, c_MGC=0.25
    )

    # Aero center
    ac = AeroCenter(x_ac_percent=25, Lambda_ac_deg=0)

    # Materials
    db = MaterialDatabase()
    materials = MaterialSelection.from_database(db, 'AL7075-T6', 'AL7075-T6', 'PLA')

    # Design space (small for testing)
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

    print(f"Design space: {design_space.total_combinations} combinations")

    # Run optimization
    best, optimizer = run_optimization(
        planform, flight, ac, materials, design_space,
        t_skin_mm=1.0, SF=1.5
    )

    if best:
        print(f"\n--- Best Configuration ---")
        print(f"  N_Rib = {best.config.N_Rib} (final: {best.N_Rib_final})")
        print(f"  X_FS = {best.config.X_FS_percent}%")
        print(f"  X_RS = {best.config.X_RS_percent}%")
        print(f"  d_FS = {best.config.d_FS_outer_mm} mm, t_FS = {best.config.t_FS_mm} mm")
        print(f"  d_RS = {best.config.d_RS_outer_mm} mm, t_RS = {best.config.t_RS_mm} mm")
        print(f"\n--- Masses ---")
        print(f"  Skin: {best.mass_skin*1000:.2f} g")
        print(f"  FS: {best.mass_FS*1000:.2f} g")
        print(f"  RS: {best.mass_RS*1000:.2f} g")
        print(f"  Ribs: {best.mass_ribs*1000:.2f} g")
        print(f"  Total: {best.mass_total*1000:.2f} g")
        print(f"\n--- Stresses ---")
        print(f"  tau_skin_max = {best.tau_skin_max/1e6:.4f} MPa")
        print(f"  sigma_vm_FS = {best.sigma_vm_FS_max/1e6:.2f} MPa")
        print(f"  sigma_vm_RS = {best.sigma_vm_RS_max/1e6:.2f} MPa")
        print(f"  sigma_skin_comp_max = {best.sigma_skin_comp_max/1e6:.4f} MPa")
        print(f"  delta_tip = {best.delta_tip*1000:.2f} mm")
        print(f"\n--- Box Beam ---")
        print(f"  I_box_root = {best.I_box_root*1e12:.2f} mm^4")
        print(f"  I_FS = {best.I_FS*1e12:.2f} mm^4")
        print(f"  I_RS = {best.I_RS*1e12:.2f} mm^4")
        print(f"  Skin I fraction = {best.skin_fraction*100:.1f}%")
        print(f"\n--- Buckling ---")
        print(f"  tau_cr_min = {best.tau_cr_min/1e6:.4f} MPa")
        print(f"  sigma_cr_min = {best.sigma_cr_min/1e6:.4f} MPa")
        print(f"  R_combined_max = {best.R_combined_max:.4f}")
        print(f"  Rib feasible: {best.rib_feasible}")

    print(f"\n--- Rejection Summary ---")
    for reason, count in optimizer.get_rejection_summary().items():
        print(f"  {reason}: {count}")

    print(f"\n--- Top 5 Solutions ---")
    for i, sol in enumerate(optimizer.get_top_solutions(5), 1):
        print(f"  {i}. mass={sol.mass_total*1000:.2f}g, N_Rib={sol.config.N_Rib}"
              f"(final:{sol.N_Rib_final}), "
              f"d_FS={sol.config.d_FS_outer_mm}mm, R_max={sol.R_combined_max:.3f}")

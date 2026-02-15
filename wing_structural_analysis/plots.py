"""
plots.py - Visualization Module

Generates all required plots for load distribution, torsion,
shear flow, and stress analysis.

Uses Matplotlib for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 150
    save_plots: bool = True
    show_plots: bool = True
    output_dir: str = "plots"
    style: str = "seaborn-v0_8-whitegrid"


def setup_plot_style(config: PlotConfig):
    """Setup matplotlib style."""
    try:
        plt.style.use(config.style)
    except:
        plt.style.use('default')

    plt.rcParams['figure.figsize'] = config.figsize
    plt.rcParams['figure.dpi'] = config.dpi
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def ensure_output_dir(config: PlotConfig):
    """Create output directory if it doesn't exist."""
    if config.save_plots and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)


def save_and_show(fig, filename: str, config: PlotConfig):
    """Save and/or show figure based on config."""
    if config.save_plots:
        ensure_output_dir(config)
        filepath = os.path.join(config.output_dir, filename)
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        print(f"  Saved: {filepath}")

    if config.show_plots:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# LOAD DISTRIBUTION PLOTS
# =============================================================================

def plot_lift_distribution(y: np.ndarray, w: np.ndarray,
                           config: PlotConfig = PlotConfig()):
    """
    Plot lift distribution w(y).

    Args:
        y: Spanwise positions [m]
        w: Distributed load [N/m]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(y * 1000, w, 'b-', linewidth=2, label='w(y)')
    ax.fill_between(y * 1000, 0, w, alpha=0.3)

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Lift Distribution w [N/m]')
    ax.set_title('Lift Distribution Along Half-Span')
    ax.legend()
    ax.set_xlim([0, y[-1] * 1000])
    ax.set_ylim(bottom=0)

    save_and_show(fig, 'lift_distribution.png', config)


def plot_shear_force(y: np.ndarray, V: np.ndarray,
                     config: PlotConfig = PlotConfig()):
    """
    Plot shear force distribution V(y).

    Args:
        y: Spanwise positions [m]
        V: Shear force [N]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(y * 1000, V, 'r-', linewidth=2, label='V(y)')
    ax.fill_between(y * 1000, 0, V, alpha=0.3, color='red')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Shear Force V [N]')
    ax.set_title('Shear Force Distribution Along Half-Span')
    ax.legend()
    ax.set_xlim([0, y[-1] * 1000])

    save_and_show(fig, 'shear_force.png', config)


def plot_bending_moment(y: np.ndarray, M: np.ndarray,
                        config: PlotConfig = PlotConfig()):
    """
    Plot bending moment distribution M(y).

    Args:
        y: Spanwise positions [m]
        M: Bending moment [N·m]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(y * 1000, M, 'g-', linewidth=2, label='M(y)')
    ax.fill_between(y * 1000, 0, M, alpha=0.3, color='green')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Bending Moment M [N·m]')
    ax.set_title('Bending Moment Distribution Along Half-Span')
    ax.legend()
    ax.set_xlim([0, y[-1] * 1000])

    save_and_show(fig, 'bending_moment.png', config)


def plot_load_distributions_combined(y: np.ndarray, w: np.ndarray,
                                      V: np.ndarray, M: np.ndarray,
                                      config: PlotConfig = PlotConfig()):
    """
    Combined plot of w(y), V(y), M(y).

    Args:
        y: Spanwise positions [m]
        w: Distributed load [N/m]
        V: Shear force [N]
        M: Bending moment [N·m]
        config: Plot configuration
    """
    fig, axes = plt.subplots(3, 1, figsize=(config.figsize[0], config.figsize[1] * 1.5))

    y_mm = y * 1000

    # w(y)
    axes[0].plot(y_mm, w, 'b-', linewidth=2)
    axes[0].fill_between(y_mm, 0, w, alpha=0.3)
    axes[0].set_ylabel('w [N/m]')
    axes[0].set_title('Load Distributions Along Half-Span')
    axes[0].set_xlim([0, y_mm[-1]])

    # V(y)
    axes[1].plot(y_mm, V, 'r-', linewidth=2)
    axes[1].fill_between(y_mm, 0, V, alpha=0.3, color='red')
    axes[1].set_ylabel('V [N]')
    axes[1].set_xlim([0, y_mm[-1]])

    # M(y)
    axes[2].plot(y_mm, M, 'g-', linewidth=2)
    axes[2].fill_between(y_mm, 0, M, alpha=0.3, color='green')
    axes[2].set_ylabel('M [N·m]')
    axes[2].set_xlabel('Spanwise Position y [mm]')
    axes[2].set_xlim([0, y_mm[-1]])

    plt.tight_layout()
    save_and_show(fig, 'load_distributions_combined.png', config)


# =============================================================================
# TORSION AND SHEAR FLOW PLOTS
# =============================================================================

def plot_torsion(y: np.ndarray, T: np.ndarray,
                 config: PlotConfig = PlotConfig()):
    """
    Plot torsion distribution T(y).

    Args:
        y: Spanwise positions [m]
        T: Torsion [N·m]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(y * 1000, T, 'm-', linewidth=2, label='T(y)')
    ax.fill_between(y * 1000, 0, T, alpha=0.3, color='magenta')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Torsion T [N·m]')
    ax.set_title('Torsion Distribution Along Half-Span')
    ax.legend()
    ax.set_xlim([0, y[-1] * 1000])
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    save_and_show(fig, 'torsion.png', config)


def plot_shear_flow(y: np.ndarray, q: np.ndarray,
                    config: PlotConfig = PlotConfig()):
    """
    Plot shear flow distribution q(y).

    Args:
        y: Spanwise positions [m]
        q: Shear flow [N/m]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(y * 1000, q, 'c-', linewidth=2, marker='o', markersize=4, label='q(y)')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Shear Flow q [N/m]')
    ax.set_title('Shear Flow Distribution Along Half-Span')
    ax.legend()
    ax.set_xlim([0, y[-1] * 1000])
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    save_and_show(fig, 'shear_flow.png', config)


def plot_skin_shear_stress(y: np.ndarray, tau_skin: np.ndarray,
                           tau_allow: Optional[float] = None,
                           config: PlotConfig = PlotConfig()):
    """
    Plot skin shear stress distribution tau_skin(y).

    Args:
        y: Spanwise positions [m]
        tau_skin: Skin shear stress [Pa]
        tau_allow: Allowable shear stress [Pa] (optional, for reference line)
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    tau_mpa = np.abs(tau_skin) / 1e6

    ax.plot(y * 1000, tau_mpa, 'orange', linewidth=2, marker='s', markersize=4, label='|τ_skin(y)|')

    if tau_allow is not None:
        ax.axhline(y=tau_allow / 1e6, color='r', linestyle='--', linewidth=2,
                   label=f'τ_allow = {tau_allow/1e6:.1f} MPa')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Skin Shear Stress |τ| [MPa]')
    ax.set_title('Skin Shear Stress Distribution Along Half-Span')
    ax.legend()
    ax.set_xlim([0, y[-1] * 1000])
    ax.set_ylim(bottom=0)

    save_and_show(fig, 'skin_shear_stress.png', config)


def plot_torsion_combined(y: np.ndarray, T: np.ndarray,
                          q: np.ndarray, tau_skin: np.ndarray,
                          tau_allow: Optional[float] = None,
                          config: PlotConfig = PlotConfig()):
    """
    Combined plot of T(y), q(y), tau_skin(y).

    Args:
        y: Spanwise positions [m]
        T: Torsion [N·m]
        q: Shear flow [N/m]
        tau_skin: Skin shear stress [Pa]
        tau_allow: Allowable shear stress [Pa]
        config: Plot configuration
    """
    fig, axes = plt.subplots(3, 1, figsize=(config.figsize[0], config.figsize[1] * 1.5))

    y_mm = y * 1000

    # T(y)
    axes[0].plot(y_mm, T, 'm-', linewidth=2)
    axes[0].fill_between(y_mm, 0, T, alpha=0.3, color='magenta')
    axes[0].set_ylabel('T [N·m]')
    axes[0].set_title('Torsion and Shear Flow Along Half-Span')
    axes[0].set_xlim([0, y_mm[-1]])
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # q(y)
    axes[1].plot(y_mm, q, 'c-', linewidth=2, marker='o', markersize=3)
    axes[1].set_ylabel('q [N/m]')
    axes[1].set_xlim([0, y_mm[-1]])
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # tau_skin(y)
    tau_mpa = np.abs(tau_skin) / 1e6
    axes[2].plot(y_mm, tau_mpa, 'orange', linewidth=2, marker='s', markersize=3)
    if tau_allow is not None:
        axes[2].axhline(y=tau_allow / 1e6, color='r', linestyle='--',
                        label=f'τ_allow = {tau_allow/1e6:.1f} MPa')
        axes[2].legend()
    axes[2].set_ylabel('|τ_skin| [MPa]')
    axes[2].set_xlabel('Spanwise Position y [mm]')
    axes[2].set_xlim([0, y_mm[-1]])
    axes[2].set_ylim(bottom=0)

    plt.tight_layout()
    save_and_show(fig, 'torsion_combined.png', config)


# =============================================================================
# SPAR STRESS PLOTS
# =============================================================================

def plot_spar_von_mises(y: np.ndarray,
                        sigma_vm_FS: np.ndarray,
                        sigma_vm_RS: np.ndarray,
                        sigma_allow: Optional[float] = None,
                        config: PlotConfig = PlotConfig()):
    """
    Plot von Mises stress in front and rear spars.

    Args:
        y: Spanwise positions [m]
        sigma_vm_FS: Von Mises stress in front spar [Pa]
        sigma_vm_RS: Von Mises stress in rear spar [Pa]
        sigma_allow: Allowable stress [Pa]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    y_mm = y * 1000

    ax.plot(y_mm, sigma_vm_FS / 1e6, 'b-', linewidth=2, marker='o',
            markersize=4, label='σ_vm Front Spar')
    ax.plot(y_mm, sigma_vm_RS / 1e6, 'r-', linewidth=2, marker='s',
            markersize=4, label='σ_vm Rear Spar')

    if sigma_allow is not None:
        ax.axhline(y=sigma_allow / 1e6, color='k', linestyle='--', linewidth=2,
                   label=f'σ_allow = {sigma_allow/1e6:.1f} MPa')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Von Mises Stress σ_vm [MPa]')
    ax.set_title('Spar Von Mises Stress Distribution')
    ax.legend()
    ax.set_xlim([0, y_mm[-1]])
    ax.set_ylim(bottom=0)

    save_and_show(fig, 'spar_von_mises.png', config)


# =============================================================================
# TWIST RATE PLOT (OPTIONAL)
# =============================================================================

def plot_twist_rate(y: np.ndarray, twist_rate: np.ndarray,
                    config: PlotConfig = PlotConfig()):
    """
    Plot twist rate dθ/dz along span.

    Args:
        y: Spanwise positions [m]
        twist_rate: Twist rate [rad/m]
        config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    twist_deg_per_m = np.degrees(twist_rate)

    ax.plot(y * 1000, twist_deg_per_m, 'purple', linewidth=2, marker='d', markersize=4)

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Twist Rate dθ/dz [deg/m]')
    ax.set_title('Twist Rate Distribution Along Half-Span')
    ax.set_xlim([0, y[-1] * 1000])

    save_and_show(fig, 'twist_rate.png', config)


# =============================================================================
# GEOMETRY VISUALIZATION
# =============================================================================

def _draw_planform_on_ax(ax, y: np.ndarray, chord: np.ndarray,
                         x_FS: np.ndarray, x_RS: np.ndarray,
                         title: str, show_legend: bool = True):
    """Draw a single planform view on the given axes.

    Helper used by plot_planform to draw both Phase-1 and Phase-2 panels.

    Args:
        ax: Matplotlib axes to draw on.
        y: Spanwise positions [m] (rib station locations).
        chord: Chord at each station [m].
        x_FS: Front spar position [m].
        x_RS: Rear spar position [m].
        title: Subplot title.
        show_legend: Whether to show legend.
    """
    y_mm = y * 1000
    N_Rib = len(y) - 1

    ax.plot(y_mm, np.zeros_like(y), 'darkgreen', linewidth=2,
            label='Leading Edge')
    ax.plot(y_mm, chord * 1000, 'darkorange', linewidth=2,
            label='Trailing Edge')
    ax.plot(y_mm, x_FS * 1000, 'b--', linewidth=2, label='Front Spar')
    ax.plot(y_mm, x_RS * 1000, 'r--', linewidth=2, label='Rear Spar')

    for i in range(len(y)):
        y_pos = y_mm[i]
        ax.plot([y_pos, y_pos], [0, chord[i] * 1000], 'g-', linewidth=1.5,
                alpha=0.7, label='Ribs' if i == 0 else None)

    ax.fill_between(y_mm, x_FS * 1000, x_RS * 1000, alpha=0.2, color='gray',
                    label='Wing Box')

    ax.set_xlabel('Spanwise Position y [mm]')
    ax.set_ylabel('Chordwise Position x [mm]')
    ax.set_title(f'{title} - {N_Rib} Bays, {len(y)} Stations')
    if show_legend:
        ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim([0, y_mm[-1]])
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()


def plot_planform(y: np.ndarray, chord: np.ndarray,
                  x_FS: np.ndarray, x_RS: np.ndarray,
                  config: PlotConfig = PlotConfig(),
                  y_phase2: Optional[np.ndarray] = None,
                  chord_at_y_func=None,
                  x_FS_at_y_func=None,
                  x_RS_at_y_func=None):
    """
    Plot wing planform with spar positions and ribs.

    If Phase-2 rib positions are provided, draws two subplots side by side:
    left = Phase-1 (uniform spacing), right = Phase-2 (adaptive/sweep).

    Args:
        y: Phase-1 spanwise positions [m] (uniform rib stations).
        chord: Chord at each Phase-1 station [m].
        x_FS: Front spar position at each Phase-1 station [m].
        x_RS: Rear spar position at each Phase-1 station [m].
        config: Plot configuration.
        y_phase2: Phase-2 rib positions [m] (adaptive). If None, single plot.
        chord_at_y_func: Callable(y)->chord [m] for interpolating Phase-2 geometry.
        x_FS_at_y_func: Callable(y)->x_FS [m] for interpolating Phase-2 geometry.
        x_RS_at_y_func: Callable(y)->x_RS [m] for interpolating Phase-2 geometry.
    """
    has_phase2 = (y_phase2 is not None and len(y_phase2) > 2 and
                  chord_at_y_func is not None)

    if has_phase2:
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                        figsize=(config.figsize[0] * 1.8,
                                                 config.figsize[1]))

        # Phase-1 (left)
        _draw_planform_on_ax(ax1, y, chord, x_FS, x_RS,
                             'Phase 1 (Uniform)', show_legend=True)

        # Phase-2 (right) — compute geometry at adaptive rib positions
        chord_p2 = np.array([chord_at_y_func(yi) for yi in y_phase2])
        x_FS_p2 = (np.array([x_FS_at_y_func(yi) for yi in y_phase2])
                    if x_FS_at_y_func is not None
                    else np.interp(y_phase2, y, x_FS))
        x_RS_p2 = (np.array([x_RS_at_y_func(yi) for yi in y_phase2])
                    if x_RS_at_y_func is not None
                    else np.interp(y_phase2, y, x_RS))

        _draw_planform_on_ax(ax2, y_phase2, chord_p2, x_FS_p2, x_RS_p2,
                             'Phase 2 (Adaptive)', show_legend=True)

        fig.suptitle('Wing Planform Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
    else:
        # Single plot (no Phase-2 data)
        fig, ax = plt.subplots(figsize=config.figsize)
        N_Rib = len(y) - 1
        _draw_planform_on_ax(ax, y, chord, x_FS, x_RS,
                             f'Wing Planform (Top View)')

    save_and_show(fig, 'planform.png', config)


# =============================================================================
# GENERATE ALL PLOTS
# =============================================================================

def generate_all_plots(y: np.ndarray, loads_data: dict,
                       torsion_data: dict, spar_data: dict,
                       geometry_data: dict, allowables: dict,
                       config: PlotConfig = PlotConfig(),
                       phase2_data: Optional[dict] = None):
    """
    Generate all required plots.

    Args:
        y: Spanwise positions [m]
        loads_data: Dict with w, V, M
        torsion_data: Dict with T, q, tau_skin, twist_rate
        spar_data: Dict with sigma_vm_FS, sigma_vm_RS
        geometry_data: Dict with chord, x_FS, x_RS
        allowables: Dict with tau_allow_skin, sigma_allow_spar
        config: Plot configuration
        phase2_data: Optional dict with Phase-2 rib data for planform comparison.
            Keys: 'y_phase2' (np.ndarray), 'chord_at_y' (callable),
                  'x_FS_at_y' (callable), 'x_RS_at_y' (callable).
    """
    print("\nGenerating plots...")
    setup_plot_style(config)

    # Load distribution plots
    plot_load_distributions_combined(
        y, loads_data['w'], loads_data['V'], loads_data['M'], config)

    # Torsion and shear flow plots
    plot_torsion_combined(
        y, torsion_data['T'], torsion_data['q'], torsion_data['tau_skin'],
        allowables.get('tau_allow_skin'), config)

    # Spar stress plot
    if 'sigma_vm_FS' in spar_data:
        plot_spar_von_mises(
            y, spar_data['sigma_vm_FS'], spar_data['sigma_vm_RS'],
            allowables.get('sigma_allow_spar'), config)

    # Twist rate (optional)
    if 'twist_rate' in torsion_data:
        plot_twist_rate(y, torsion_data['twist_rate'], config)

    # Planform (with optional Phase-2 comparison)
    p2_kwargs = {}
    if phase2_data is not None:
        p2_kwargs = {
            'y_phase2': phase2_data.get('y_phase2'),
            'chord_at_y_func': phase2_data.get('chord_at_y'),
            'x_FS_at_y_func': phase2_data.get('x_FS_at_y'),
            'x_RS_at_y_func': phase2_data.get('x_RS_at_y'),
        }
    plot_planform(y, geometry_data['chord'], geometry_data['x_FS'],
                  geometry_data['x_RS'], config, **p2_kwargs)

    print("All plots generated.")


if __name__ == "__main__":
    print("=== Plots Module Test ===\n")

    # Create test data
    y = np.linspace(0, 1, 11)  # 0 to 1m
    w = 50 * np.sqrt(1 - (y / 1)**2 + 0.01)  # Elliptic-ish
    V = np.array([np.trapz(w[i:], y[i:]) for i in range(len(y))])
    M = np.array([np.trapz(V[i:], y[i:]) for i in range(len(y))])
    T = -0.5 * M  # Fake torsion
    q = T / (2 * 0.005)  # Fake shear flow
    tau_skin = q / 0.001  # Fake stress
    twist_rate = q / (2 * 0.005 * 27e9)  # Fake twist

    # Geometry
    chord = 0.3 - 0.15 * y
    x_FS = 0.25 * chord
    x_RS = 0.65 * chord

    # Spar stress (fake)
    sigma_vm_FS = 30e6 * (1 - y)
    sigma_vm_RS = 25e6 * (1 - y)

    # Config - don't show, just save
    config = PlotConfig(
        save_plots=True,
        show_plots=False,
        output_dir="test_plots"
    )

    # Generate all plots
    generate_all_plots(
        y=y,
        loads_data={'w': w, 'V': V, 'M': M},
        torsion_data={'T': T, 'q': q, 'tau_skin': tau_skin, 'twist_rate': twist_rate},
        spar_data={'sigma_vm_FS': sigma_vm_FS, 'sigma_vm_RS': sigma_vm_RS},
        geometry_data={'chord': chord, 'x_FS': x_FS, 'x_RS': x_RS},
        allowables={'tau_allow_skin': 200e6, 'sigma_allow_spar': 380e6},
        config=config
    )

    print("\nTest completed. Check 'test_plots' directory.")

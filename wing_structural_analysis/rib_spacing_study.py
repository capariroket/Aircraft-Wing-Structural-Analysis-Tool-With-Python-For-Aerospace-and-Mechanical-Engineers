"""
rib_spacing_study.py - Rib Spacing vs Skin Buckling Trade Study

Bu script:
1. Default parametrelerle geometri hesaplar (b, C_r, c_MGC, N_Rib_min/max)
2. Farkli rib spacing'lerde skin compression buckling sigma_cr hesaplar
3. t_skin = 1.2mm icin gereken minimum rib spacing'i bulur
4. Farkli t_skin degerleri icin sigma_cr karsilastirmasi yapar
5. Kutle trade-off analizi yapar: ince skin + cok rib vs kalin skin + az rib

Formul:
  sigma_cr = k_c * pi^2 * E / (12*(1-nu^2)) * (t/s)^2
  k_c = 4.0 (basit mesnetli duz plaka, uniform basinc)
"""

import math

# =============================================================================
# 1) INPUT PARAMETERS
# =============================================================================
AR = 11.0
S_ref = 3.17  # m^2
lam = 0.45    # taper ratio
t_over_c = 0.17
n = 2.0       # load factor
W_0 = 650.0   # N (MTOW)

# Skin material: PLA
E_skin = 3.5e9    # Pa
nu_skin = 0.36
rho_skin = 1200.0  # kg/m^3

# Spar material: CFRP
E_spar = 135e9     # Pa
rho_spar = 1600.0  # kg/m^3

# Buckling coefficient
k_c = 4.0  # simply supported, uniform compression

# Skin thickness (design variable)
t_skin_mm = 1.2
t_skin = t_skin_mm / 1000  # m

# Rib thickness (for mass calc)
t_rib_mm = 3.0
t_rib = t_rib_mm / 1000  # m
rho_rib = rho_skin  # PLA ribs

print("=" * 72)
print("  RIB SPACING vs SKIN BUCKLING TRADE STUDY")
print("=" * 72)

# =============================================================================
# 2) GEOMETRY CALCULATIONS
# =============================================================================
b = math.sqrt(AR * S_ref)
C_r = 2 * S_ref / (b * (1 + lam))
c_MGC = (4.0 / 3.0) * math.sqrt(S_ref / AR) * ((1 + lam + lam**2) / (1 + lam)**2)
Y_bar_mm = (b / 6) * ((1 + 2 * lam) / (1 + lam)) * 1000  # mm
Y_bar = Y_bar_mm / 1000  # m
c_tip = lam * C_r
L_span = b / 2

N_Rib_min = math.ceil(1 + b / c_MGC)
N_Rib_max = 2 * N_Rib_min

spacing_min = L_span / N_Rib_min  # spacing with N_Rib_min ribs
spacing_max_ribs = L_span / N_Rib_max  # spacing with N_Rib_max ribs

print("\n--- PLANFORM GEOMETRY ---")
print(f"  b (wingspan)     = {b:.4f} m  ({b*1000:.1f} mm)")
print(f"  C_r (root chord) = {C_r:.4f} m  ({C_r*1000:.1f} mm)")
print(f"  c_tip            = {c_tip:.4f} m  ({c_tip*1000:.1f} mm)")
print(f"  c_MGC            = {c_MGC:.4f} m  ({c_MGC*1000:.1f} mm)")
print(f"  Y_bar            = {Y_bar:.4f} m  ({Y_bar_mm:.1f} mm)")
print(f"  L_span (half)    = {L_span:.4f} m  ({L_span*1000:.1f} mm)")
print(f"  N_Rib_min        = {N_Rib_min}")
print(f"  N_Rib_max        = {N_Rib_max}")
print(f"  Spacing (N_Rib_min = {N_Rib_min} ribs) = {spacing_min*1000:.1f} mm")
print(f"  Spacing (N_Rib_max = {N_Rib_max} ribs) = {spacing_max_ribs*1000:.1f} mm")

# =============================================================================
# 3) BUCKLING CRITICAL STRESS vs RIB SPACING
# =============================================================================
def sigma_cr_buckling(k, E, nu, t, s):
    """
    Flat plate buckling critical stress.
    sigma_cr = k * pi^2 * E / (12*(1-nu^2)) * (t/s)^2

    Args:
        k: buckling coefficient (4.0 for compression, simply supported)
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        t: plate thickness [m]
        s: unsupported length (rib spacing) [m]
    Returns:
        Critical buckling stress [Pa]
    """
    return k * math.pi**2 * E / (12 * (1 - nu**2)) * (t / s)**2

print("\n" + "=" * 72)
print("  SKIN COMPRESSION BUCKLING: sigma_cr vs RIB SPACING")
print(f"  t_skin = {t_skin_mm} mm, E = {E_skin/1e9} GPa, nu = {nu_skin}, k_c = {k_c}")
print("=" * 72)

# Spacing values to check
spacings_mm = [
    spacing_min * 1000,          # N_Rib_min spacing
    spacing_max_ribs * 1000,     # N_Rib_max spacing
    20, 25, 30, 40, 50, 100
]

# Applied skin compressive stress candidates
sigma_comp_low = 20e6   # 20 MPa
sigma_comp_high = 30e6  # 30 MPa

print(f"\n  {'Spacing [mm]':>14} | {'sigma_cr [MPa]':>14} | {'sigma_cr/20MPa':>14} | {'sigma_cr/30MPa':>14} | {'Status (30MPa)':>16}")
print("  " + "-" * 80)

for s_mm in sorted(set(spacings_mm)):
    s = s_mm / 1000
    scr = sigma_cr_buckling(k_c, E_skin, nu_skin, t_skin, s)
    scr_MPa = scr / 1e6
    ratio_low = scr / sigma_comp_low
    ratio_high = scr / sigma_comp_high
    status = "OK (>=1)" if ratio_high >= 1.0 else "FAIL (<1)"

    label = ""
    if abs(s_mm - spacing_min * 1000) < 0.1:
        label = f" [N_Rib_min={N_Rib_min}]"
    elif abs(s_mm - spacing_max_ribs * 1000) < 0.1:
        label = f" [N_Rib_max={N_Rib_max}]"

    print(f"  {s_mm:>10.1f} mm{label:20s} | {scr_MPa:14.3f} | {ratio_low:14.3f} | {ratio_high:14.3f} | {status:>16}")

# =============================================================================
# 4) REQUIRED t_skin FOR sigma_cr >= 30 MPa AT N_Rib_max SPACING
# =============================================================================
print("\n" + "=" * 72)
print("  REQUIRED t_skin FOR sigma_cr >= 30 MPa at N_Rib_max spacing")
print("=" * 72)

s_target = spacing_max_ribs  # m
sigma_target = sigma_comp_high  # 30 MPa

# Inverse formula:
# sigma_cr = k * pi^2 * E / (12*(1-nu^2)) * (t/s)^2
# t = s * sqrt(sigma_target * 12*(1-nu^2) / (k * pi^2 * E))
t_required = s_target * math.sqrt(sigma_target * 12 * (1 - nu_skin**2) / (k_c * math.pi**2 * E_skin))

print(f"\n  Target spacing      = {s_target*1000:.1f} mm (N_Rib_max = {N_Rib_max})")
print(f"  Target sigma_comp   = {sigma_target/1e6:.0f} MPa")
print(f"  Required t_skin     = {t_required*1000:.3f} mm")
print(f"  Current t_skin      = {t_skin_mm:.1f} mm")
print(f"  Ratio (required/current) = {t_required/t_skin:.2f}x")

# Verify
scr_verify = sigma_cr_buckling(k_c, E_skin, nu_skin, t_required, s_target)
print(f"  Verification: sigma_cr @ t={t_required*1000:.3f}mm, s={s_target*1000:.1f}mm = {scr_verify/1e6:.3f} MPa")

# =============================================================================
# 5) sigma_cr FOR DIFFERENT t_skin VALUES AT N_Rib_max SPACING
# =============================================================================
print("\n" + "=" * 72)
print(f"  sigma_cr vs t_skin at N_Rib_max spacing ({spacing_max_ribs*1000:.1f} mm)")
print("=" * 72)

t_skin_values_mm = [0.625, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

print(f"\n  {'t_skin [mm]':>12} | {'sigma_cr [MPa]':>14} | {'ratio vs 20MPa':>14} | {'ratio vs 30MPa':>14} | {'Status (30MPa)':>16}")
print("  " + "-" * 80)

for t_mm in t_skin_values_mm:
    t = t_mm / 1000
    scr = sigma_cr_buckling(k_c, E_skin, nu_skin, t, spacing_max_ribs)
    scr_MPa = scr / 1e6
    r20 = scr / sigma_comp_low
    r30 = scr / sigma_comp_high
    status = "OK" if r30 >= 1.0 else "FAIL"
    marker = " <-- current" if abs(t_mm - t_skin_mm) < 0.01 else ""
    marker = " <-- required" if abs(t_mm - t_required * 1000) < 0.05 else marker
    print(f"  {t_mm:>10.3f} mm | {scr_MPa:14.3f} | {r20:14.3f} | {r30:14.3f} | {status:>16}{marker}")

# =============================================================================
# 6) ALSO: sigma_cr AT N_Rib_min SPACING FOR DIFFERENT t_skin
# =============================================================================
print("\n" + "=" * 72)
print(f"  sigma_cr vs t_skin at N_Rib_min spacing ({spacing_min*1000:.1f} mm)")
print("=" * 72)

print(f"\n  {'t_skin [mm]':>12} | {'sigma_cr [MPa]':>14} | {'ratio vs 30MPa':>14} | {'Status (30MPa)':>16}")
print("  " + "-" * 50)

for t_mm in t_skin_values_mm:
    t = t_mm / 1000
    scr = sigma_cr_buckling(k_c, E_skin, nu_skin, t, spacing_min)
    scr_MPa = scr / 1e6
    r30 = scr / sigma_comp_high
    status = "OK" if r30 >= 1.0 else "FAIL"
    print(f"  {t_mm:>10.3f} mm | {scr_MPa:14.3f} | {r30:14.3f} | {status:>16}")

# =============================================================================
# 7) MASS TRADE-OFF ANALYSIS
# =============================================================================
print("\n" + "=" * 72)
print("  MASS TRADE-OFF: thin skin + many ribs vs thick skin + few ribs")
print("=" * 72)

# Wing box geometry for rib area estimation
# Spar positions: FS = 25%, RS = 65% (typical)
X_FS_pct = 25.0
X_RS_pct = 65.0
h_box_root = t_over_c * C_r

# Approximate rib area (root, using parabolic)
# S_rib ~ (2/3) * C_r * h_box_root * (X_RS% - X_FS%)/100
# But more precisely: S_rib_total = (2/3) * x_RS * h_box - that gives LE to RS area
# For wing box rib only (FS to RS):
x_FS_root = X_FS_pct / 100 * C_r
x_RS_root = X_RS_pct / 100 * C_r
S_rib_LE_FS = (2.0/3.0) * x_FS_root * h_box_root
S_rib_LE_RS = (2.0/3.0) * x_RS_root * h_box_root
S_rib_box = S_rib_LE_RS - S_rib_LE_FS  # FS to RS section only

# For simplicity, use average rib area (root to tip linear taper)
h_box_tip = t_over_c * c_tip
x_FS_tip = X_FS_pct / 100 * c_tip
x_RS_tip = X_RS_pct / 100 * c_tip
S_rib_box_tip = (2.0/3.0) * x_RS_tip * h_box_tip - (2.0/3.0) * x_FS_tip * h_box_tip
S_rib_avg = (S_rib_box + S_rib_box_tip) / 2

print(f"\n  --- Rib geometry ---")
print(f"  h_box_root    = {h_box_root*1000:.1f} mm")
print(f"  h_box_tip     = {h_box_tip*1000:.1f} mm")
print(f"  S_rib_root    = {S_rib_box*1e6:.1f} mm^2 = {S_rib_box*1e4:.2f} cm^2")
print(f"  S_rib_tip     = {S_rib_box_tip*1e6:.1f} mm^2 = {S_rib_box_tip*1e4:.2f} cm^2")
print(f"  S_rib_avg     = {S_rib_avg*1e6:.1f} mm^2 = {S_rib_avg*1e4:.2f} cm^2")

# Skin area (half-wing, top + bottom)
S_skin_half = 2 * ((C_r + c_tip) / 2) * L_span

print(f"  S_skin_half   = {S_skin_half:.4f} m^2 = {S_skin_half*1e4:.0f} cm^2")

# Configuration A: t_skin=1.2mm, N_Rib_max ribs
t_A = 1.2e-3
N_A = N_Rib_max
skin_mass_A = S_skin_half * t_A * rho_skin
rib_mass_A = N_A * S_rib_avg * t_rib * rho_rib

# Check buckling for config A
scr_A = sigma_cr_buckling(k_c, E_skin, nu_skin, t_A, L_span / N_A)

# Configuration B: t_skin=2.0mm, N_Rib_min ribs
t_B = 2.0e-3
N_B = N_Rib_min
skin_mass_B = S_skin_half * t_B * rho_skin
rib_mass_B = N_B * S_rib_avg * t_rib * rho_rib

# Check buckling for config B
scr_B = sigma_cr_buckling(k_c, E_skin, nu_skin, t_B, L_span / N_B)

# Configuration C: t_skin=t_required, N_Rib_max ribs (exact 30 MPa match)
t_C = t_required
N_C = N_Rib_max
skin_mass_C = S_skin_half * t_C * rho_skin
rib_mass_C = N_C * S_rib_avg * t_rib * rho_rib
scr_C = sigma_cr_buckling(k_c, E_skin, nu_skin, t_C, L_span / N_C)

# Configuration D: t_skin=t_required for N_Rib_min spacing, N_Rib_min ribs
t_required_nmin = spacing_min * math.sqrt(sigma_comp_high * 12 * (1 - nu_skin**2) / (k_c * math.pi**2 * E_skin))
t_D = t_required_nmin
N_D = N_Rib_min
skin_mass_D = S_skin_half * t_D * rho_skin
rib_mass_D = N_D * S_rib_avg * t_rib * rho_rib
scr_D = sigma_cr_buckling(k_c, E_skin, nu_skin, t_D, L_span / N_D)

print(f"\n  --- CONFIGURATION COMPARISON (half-wing, skin + ribs only) ---")
print(f"  {'Config':>8} | {'t_skin[mm]':>10} | {'N_Rib':>6} | {'spacing[mm]':>11} | {'sigma_cr[MPa]':>13} | {'Buckling':>10} | {'m_skin[g]':>10} | {'m_ribs[g]':>10} | {'m_total[g]':>11}")
print("  " + "-" * 115)

configs = [
    ("A", t_A, N_A, scr_A, skin_mass_A, rib_mass_A),
    ("B", t_B, N_B, scr_B, skin_mass_B, rib_mass_B),
    ("C", t_C, N_C, scr_C, skin_mass_C, rib_mass_C),
    ("D", t_D, N_D, scr_D, skin_mass_D, rib_mass_D),
]

for label, t_val, n_rib, scr, m_skin, m_rib in configs:
    spacing = L_span / n_rib
    status = "OK" if scr >= sigma_comp_high else "FAIL"
    m_total = m_skin + m_rib
    desc = ""
    if label == "A":
        desc = "  <- 1.2mm + max ribs"
    elif label == "B":
        desc = "  <- 2.0mm + min ribs"
    elif label == "C":
        desc = "  <- exact 30MPa + max ribs"
    elif label == "D":
        desc = "  <- exact 30MPa + min ribs"
    print(f"  {label:>8} | {t_val*1000:>10.3f} | {n_rib:>6} | {spacing*1000:>11.1f} | {scr/1e6:>13.3f} | {status:>10} | {m_skin*1000:>10.1f} | {m_rib*1000:>10.1f} | {m_total*1000:>11.1f}{desc}")

print(f"\n  --- SUMMARY ---")
winner_AB = "A" if (skin_mass_A + rib_mass_A) < (skin_mass_B + rib_mass_B) else "B"
m_A_total = (skin_mass_A + rib_mass_A) * 1000
m_B_total = (skin_mass_B + rib_mass_B) * 1000
m_C_total = (skin_mass_C + rib_mass_C) * 1000
m_D_total = (skin_mass_D + rib_mass_D) * 1000

print(f"  Config A (t=1.2mm, N={N_Rib_max}):    {m_A_total:.1f} g  sigma_cr={sigma_cr_buckling(k_c, E_skin, nu_skin, t_A, L_span/N_A)/1e6:.2f} MPa")
print(f"  Config B (t=2.0mm, N={N_Rib_min}):    {m_B_total:.1f} g  sigma_cr={sigma_cr_buckling(k_c, E_skin, nu_skin, t_B, L_span/N_B)/1e6:.2f} MPa")
print(f"  Config C (t={t_required*1000:.2f}mm, N={N_Rib_max}): {m_C_total:.1f} g  sigma_cr=30.00 MPa (exact)")
print(f"  Config D (t={t_required_nmin*1000:.2f}mm, N={N_Rib_min}): {m_D_total:.1f} g  sigma_cr=30.00 MPa (exact)")

print(f"\n  Mass difference A vs B: {abs(m_A_total - m_B_total):.1f} g, winner = Config {winner_AB}")
all_totals = {"A": m_A_total, "B": m_B_total, "C": m_C_total, "D": m_D_total}
lightest = min(all_totals, key=all_totals.get)
print(f"  Lightest feasible config: {lightest} ({all_totals[lightest]:.1f} g)")

# Buckling feasibility check
print(f"\n  --- BUCKLING FEASIBILITY ---")
for label, t_val, n_rib, scr, m_skin, m_rib in configs:
    spacing = L_span / n_rib
    scr_val = scr / 1e6
    margin = scr / sigma_comp_high - 1.0
    feasible = "FEASIBLE" if scr >= sigma_comp_high else "NOT FEASIBLE"
    print(f"  Config {label}: sigma_cr = {scr_val:.2f} MPa, margin = {margin*100:+.1f}%, {feasible}")

print("\n" + "=" * 72)
print("  DONE")
print("=" * 72)

#!/usr/bin/env python3
"""
verify_results.py - Bağımsız Sonuç Doğrulama Scripti

Proje modüllerini import etmeden, sadece math/numpy ile
optimization_results.json'daki her değeri sıfırdan hesaplar.
Her adımda "Beklenen vs Hesaplanan vs Fark" yazdırır.

Kullanım:
    python3 verify_results.py
"""

import json
import math
import numpy as np

# =============================================================================
# JSON ÇIKTIYI OKU
# =============================================================================

with open("optimization_results.json", "r") as f:
    data = json.load(f)

cfg = data["optimal_config"]
mass = data["mass_breakdown"]
stress = data["stress_results"]
reactions = data["root_reactions"]
box = data["box_beam"]
buckling = data["buckling"]

# =============================================================================
# GİRDİ PARAMETRELERİ (Default değerler - hardcoded)
# =============================================================================

# Flight science
C_m = -0.003
AR = 11.0
lam = 0.45          # taper ratio
V_c = 21.0          # m/s
n_load = 2.0        # load factor
S_ref = 3.17        # m²
W_0 = 650.0         # N
rho_air = 1.223     # kg/m³
x_ac_pct = 25.0     # %
Lambda_ac_deg = 0.0  # deg

# Geometry
b = 5.195           # m
t_over_c = 0.17

# Structural
t_skin_mm = 1.2
SF = 1.0

# Best config from JSON
N_Rib = cfg["N_Rib"]
t_rib_mm = cfg["t_rib_mm"]
X_FS_pct = cfg["X_FS_percent"]
X_RS_pct = cfg["X_RS_percent"]
d_FS_mm = cfg["d_FS_outer_mm"]
t_FS_mm = cfg["t_FS_mm"]
d_RS_mm = cfg["d_RS_outer_mm"]
t_RS_mm = cfg["t_RS_mm"]

# Material properties (hardcoded from database)
# CFRP (spar)
E_spar = 135.0e9    # Pa
G_spar = 5.0e9      # Pa
sigma_u_spar = 1500e6  # Pa
tau_u_spar = 80e6    # Pa (interlaminar)
rho_spar = 1600.0    # kg/m³

# PLA (skin & rib)
E_skin = 3.5e9       # Pa
G_skin = 1.287e9     # Pa
sigma_u_skin = 50e6  # Pa
tau_u_skin = 30e6    # Pa (actually tau_u = sigma_u * 0.6 = 30 MPa)
rho_skin = 1250.0    # kg/m³
nu_skin = 0.36

E_rib = E_skin
rho_rib = rho_skin

# =============================================================================
# DOĞRULAMA FONKSİYONLARI
# =============================================================================

pass_count = 0
fail_count = 0
warn_count = 0

def check(name, expected, computed, tol_pct=0.5, unit=""):
    """Bir değeri karşılaştır. tol_pct = yüzde tolerans."""
    global pass_count, fail_count, warn_count

    if expected == 0 and computed == 0:
        status = "PASS"
        pct = 0.0
    elif expected == 0:
        pct = abs(computed) * 100
        status = "WARN" if abs(computed) < 1e-10 else "FAIL"
    else:
        pct = abs(computed - expected) / abs(expected) * 100
        if pct <= tol_pct:
            status = "PASS"
        elif pct <= 2.0:
            status = "WARN"
        else:
            status = "FAIL"

    if status == "PASS":
        pass_count += 1
        marker = "  ✓"
    elif status == "WARN":
        warn_count += 1
        marker = "  ⚠"
    else:
        fail_count += 1
        marker = "  ✗"

    print(f"{marker} {name:40s} | Beklenen: {expected:>14.6f} | Hesaplanan: {computed:>14.6f} | Δ: {pct:>6.2f}% {unit}")
    return status

# =============================================================================
# BÖLÜM 1: GEOMETRİ HESAPLARI
# =============================================================================

print("=" * 110)
print("BÖLÜM 1: GEOMETRİ DOĞRULAMASI")
print("=" * 110)

# C_r = 2 * S_ref / (b * (1 + λ))
C_r = 2 * S_ref / (b * (1 + lam))    # [m]
C_r_mm = C_r * 1000

# c_MGC = (4/3) * sqrt(S_ref/AR) * ((1+λ+λ²)/(1+λ)²)
c_MGC = (4/3) * math.sqrt(S_ref / AR) * ((1 + lam + lam**2) / (1 + lam)**2)

# Ȳ = (b/6) * ((1+2λ)/(1+λ)) * 1000  [mm]
Y_bar_mm = (b / 6) * ((1 + 2*lam) / (1 + lam)) * 1000

# N_Rib = ceil(1 + sqrt(AR*S_ref) / c_MGC)
N_Rib_calc = math.ceil(1 + math.sqrt(AR * S_ref) / c_MGC)

L_span = b / 2
c_tip = lam * C_r
h_box_root = t_over_c * C_r

print(f"  C_r = {C_r_mm:.4f} mm")
print(f"  c_MGC = {c_MGC*1000:.4f} mm")
print(f"  Y_bar = {Y_bar_mm:.4f} mm")
print(f"  N_Rib = {N_Rib_calc}")
print(f"  L_span = {L_span:.4f} m")
print(f"  h_box_root = {h_box_root*1000:.4f} mm")

check("N_Rib", cfg["N_Rib"], N_Rib_calc)
check("rib_spacing_mm", cfg["rib_spacing_mm"], L_span / N_Rib * 1000, tol_pct=0.01)

# =============================================================================
# BÖLÜM 2: SPAR GEOMETRİSİ
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 2: SPAR GEOMETRİSİ DOĞRULAMASI")
print("=" * 110)

# Spar positions at root
X_FS_at_root = C_r * (X_FS_pct / 100)  # [m]
X_RS_at_root = C_r * (X_RS_pct / 100)  # [m]

check("X_FS_mm", cfg["X_FS_mm"], X_FS_at_root * 1000)
check("X_RS_mm", cfg["X_RS_mm"], X_RS_at_root * 1000)

# Sweep angles
# Λ_FS = arctan(tan(Λ_ac) + (4*(x_ac - X_FS%)*(1-λ)) / (x_ac*AR*(1+λ)))
# Note: x_ac is in percent here
Lambda_ac_rad = math.radians(Lambda_ac_deg)
Lambda_FS_rad = math.atan(math.tan(Lambda_ac_rad) +
    (4 * (x_ac_pct - X_FS_pct) * (1 - lam)) / (x_ac_pct * AR * (1 + lam)))
Lambda_RS_rad = math.atan(math.tan(Lambda_ac_rad) +
    (4 * (x_ac_pct - X_RS_pct) * (1 - lam)) / (x_ac_pct * AR * (1 + lam)))

Lambda_FS_deg = math.degrees(Lambda_FS_rad)
Lambda_RS_deg = math.degrees(Lambda_RS_rad)

check("Lambda_FS_deg", cfg["Lambda_FS_deg"], Lambda_FS_deg)
check("Lambda_RS_deg", cfg["Lambda_RS_deg"], Lambda_RS_deg)

# Spar lengths
# NOT: main.py raporu 0.5*sqrt(AR*S_ref)/cos(Λ) kullanıyor, b/2 değil
# b (5.195) ≠ sqrt(AR*S_ref) (5.906) olduğundan fark var
# Kütle hesabı doğru (b/2 kullanır), sadece rapordaki görüntü farklı
L_FS_actual = L_span / math.cos(Lambda_FS_rad)   # Kütle hesabında kullanılan
L_RS_actual = L_span / math.cos(Lambda_RS_rad)
L_FS_report = 0.5 * math.sqrt(AR * S_ref) / math.cos(Lambda_FS_rad)  # Rapordaki
L_RS_report = 0.5 * math.sqrt(AR * S_ref) / math.cos(Lambda_RS_rad)

check("L_FS_mm (rapor formülü)", cfg["L_FS_mm"], L_FS_report * 1000)
check("L_RS_mm (rapor formülü)", cfg["L_RS_mm"], L_RS_report * 1000)
L_FS = L_FS_actual  # kütle hesabı için doğru değeri kullan
L_RS = L_RS_actual
print(f"  ⚠ NOT: L_spar raporda sqrt(AR*S_ref)/2={L_FS_report*1000:.1f}mm, gerçekte b/2={L_FS*1000:.1f}mm")

# Load sharing
eta_FS = (X_RS_pct - x_ac_pct) / (X_RS_pct - X_FS_pct)
eta_RS = (x_ac_pct - X_FS_pct) / (X_RS_pct - X_FS_pct)

check("eta_FS", cfg["eta_FS_percent"] / 100, eta_FS)
check("eta_RS", cfg["eta_RS_percent"] / 100, eta_RS)

# Cross-section properties (circular tube)
d_FS_o = d_FS_mm / 1000  # [m]
d_FS_i = d_FS_o - 2 * (t_FS_mm / 1000)
d_RS_o = d_RS_mm / 1000
d_RS_i = d_RS_o - 2 * (t_RS_mm / 1000)

A_FS = (math.pi / 4) * (d_FS_o**2 - d_FS_i**2)  # [m²]
A_RS = (math.pi / 4) * (d_RS_o**2 - d_RS_i**2)

I_FS = (math.pi / 64) * (d_FS_o**4 - d_FS_i**4)  # [m⁴]
I_RS = (math.pi / 64) * (d_RS_o**4 - d_RS_i**4)
J_FS = 2 * I_FS
J_RS = 2 * I_RS

check("A_Act_FS_mm2", cfg["A_Act_FS_mm2"], A_FS * 1e6)
check("A_Act_RS_mm2", cfg["A_Act_RS_mm2"], A_RS * 1e6)
check("I_FS_mm4", cfg["I_FS_mm4"], I_FS * 1e12)
check("I_RS_mm4", cfg["I_RS_mm4"], I_RS * 1e12)

# Critical area: A_Cri = (n * W_0 * Ȳ_m * 0.5 * η) / (2 * σ_max * c)
Y_bar_m = Y_bar_mm / 1000
c_FS = d_FS_o / 2
c_RS = d_RS_o / 2

A_Cri_FS = (n_load * W_0 * Y_bar_m * 0.5 * eta_FS) / (2 * sigma_u_spar * c_FS)
A_Cri_RS = (n_load * W_0 * Y_bar_m * 0.5 * eta_RS) / (2 * sigma_u_spar * c_RS)

check("A_Cri_FS_mm2", cfg["A_Cri_FS_mm2"], A_Cri_FS * 1e6)
check("A_Cri_RS_mm2", cfg["A_Cri_RS_mm2"], A_Cri_RS * 1e6)

# Area adequacy check
print(f"\n  A_Act_FS > A_Cri_FS : {A_FS*1e6:.2f} > {A_Cri_FS*1e6:.2f} → {'PASS' if A_FS > A_Cri_FS else 'FAIL'}")
print(f"  A_Act_RS > A_Cri_RS : {A_RS*1e6:.2f} > {A_Cri_RS*1e6:.2f} → {'PASS' if A_RS > A_Cri_RS else 'FAIL'}")

# =============================================================================
# BÖLÜM 3: BOX BEAM (Transformed Section)
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 3: BOX BEAM DOĞRULAMASI")
print("=" * 110)

t_skin = t_skin_mm / 1000  # [m]
w_box = X_RS_at_root - X_FS_at_root  # box width at root [m]

# Modular ratio
n_ratio = E_skin / E_spar

# Skin contribution (parallel axis theorem)
d_NA = h_box_root / 2  # distance from NA to skin panel
I_skin_parallel = 2 * w_box * t_skin * d_NA**2   # Steiner term
I_skin_self = 2 * w_box * t_skin**3 / 12          # Self-inertia
I_skin_transformed = n_ratio * (I_skin_parallel + I_skin_self)

I_box = I_FS + I_RS + I_skin_transformed

check("I_box_root_mm4", box["I_box_root_mm4"], I_box * 1e12, tol_pct=0.01)
check("I_FS_mm4 (box)", box["I_FS_mm4"], I_FS * 1e12)
check("I_RS_mm4 (box)", box["I_RS_mm4"], I_RS * 1e12)

skin_frac = I_skin_transformed / I_box
check("skin_fraction", box["skin_fraction"], skin_frac, tol_pct=0.01)

print(f"\n  I_skin_parallel = {I_skin_parallel*1e12:.2f} mm⁴")
print(f"  I_skin_self     = {I_skin_self*1e12:.4f} mm⁴")
print(f"  n_ratio (E_skin/E_spar) = {n_ratio:.6f}")
print(f"  I_skin_transformed = {I_skin_transformed*1e12:.2f} mm⁴")

# =============================================================================
# BÖLÜM 4: KÜTLE HESAPLARI
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 4: KÜTLE DOĞRULAMASI")
print("=" * 110)

# Skin mass: m = 2 * ((C_r + c_tip)/2) * L_span * t_skin * ρ_skin
# This is the trapezoidal approximation for the half-wing skin area
S_skin_area = 2 * ((C_r + c_tip) / 2) * L_span  # upper + lower surface [m²]
m_skin = S_skin_area * t_skin * rho_skin

# Spar masses: m = A * L * ρ
m_FS = A_FS * L_FS * rho_spar
m_RS = A_RS * L_RS * rho_spar

# Rib mass: Need to compute per-station rib areas
# For rectangular profile: S_rib = w_box * h_box at each station
# For N_Rib ribs at uniform spacing, rib areas vary with taper
# y_ribs = i * L_span / N_Rib for i = 0..N_Rib
# chord(y) = C_r - (C_r - c_tip) * y / L_span = C_r * (1 - (1-λ)*y/L_span)
# h_box(y) = t/c * chord(y)
# x_FS(y) = (X_FS_pct/100) * chord(y), x_RS(y) = (X_RS_pct/100) * chord(y)
# w_box(y) = x_RS(y) - x_FS(y) = chord(y) * (X_RS_pct - X_FS_pct) / 100
# S_rib(y) = w_box(y) * h_box(y) = chord(y)² * t/c * (X_RS_pct - X_FS_pct) / 100

t_rib = t_rib_mm / 1000
m_ribs = 0.0
for i in range(N_Rib + 1):  # N_Rib+1 stations (0 to N_Rib inclusive = N_Rib+1 ribs??)
    # Actually N_Rib ribs... let me reconsider.
    # The code uses N_Rib = 12, which creates 13 stations (0..12)
    # But number of physical ribs = N_Rib (12 bays, N_Rib+1 stations, but ribs at each station = N_Rib+1)
    # Wait... let me think. If you have 12 ribs, you get 12+1=13 stations and 12 bays.
    # Or is it: N_Rib = number of bays, so N_Rib+1 stations with N_Rib+1 ribs?
    # From the output: "13 stations (12 bays)" and N_Rib=12
    # So there are 13 rib stations.
    pass

# Let me compute rib mass properly
# N_Rib = 12 means 12 bays, 13 rib stations (ribs at root, tip, and 11 intermediate)
# Actually the code says N_Rib is number of ribs, not bays. Let me check.
# From CLAUDE.md: N_Rib = ceil(1 + sqrt(AR*S_ref) / c_MGC)
# And the output shows "N_Rib (number of ribs) = 12" with "13 stations (12 bays)"
# So 12 ribs + root = 13 stations? Or 12+1 ribs at 13 stations?

# Actually: the stations are at y_i = i * L_span / N_Rib for i = 0..N_Rib
# That gives N_Rib+1 = 13 stations
# Each station has a rib, so there are 13 physical ribs

# Rib area = LE-to-FS + FS-to-RS (full rib from leading edge to rear spar)
# Rectangular profile: S_rib = X_RS * h_box (= (X_RS_pct/100) * chord * t/c * chord)
# NOT: Root rib (i=0) dahil EDİLMEZ (gövde parçası olarak kabul edilir)
m_ribs_calc = 0.0
for i in range(N_Rib + 1):
    y_i = i * L_span / N_Rib
    chord_i = C_r * (1 - (1 - lam) * y_i / L_span)
    h_box_i = t_over_c * chord_i
    x_RS_i = (X_RS_pct / 100) * chord_i
    S_rib_i = x_RS_i * h_box_i  # full rib area (LE to RS)
    m_rib_i = S_rib_i * t_rib * rho_rib
    if i == 0:
        print(f"  Root rib (i=0): S={S_rib_i*1e6:.1f} mm², m={m_rib_i*1000:.2f} g [EXCLUDED]")
    else:
        m_ribs_calc += m_rib_i

check("m_skin [kg]", mass["m_skin"], m_skin, tol_pct=1.0)
check("m_FS [kg]", mass["m_FS"], m_FS, tol_pct=0.5)
check("m_RS [kg]", mass["m_RS"], m_RS, tol_pct=0.5)
check("m_ribs [kg]", mass["m_ribs"], m_ribs_calc, tol_pct=2.0, unit="(per-station sum)")

m_total_calc = m_skin + m_FS + m_RS + m_ribs_calc
check("m_total [kg]", mass["m_total"], m_total_calc, tol_pct=2.0)

# Internal consistency check
m_total_json_sum = mass["m_skin"] + mass["m_FS"] + mass["m_RS"] + mass["m_ribs"]
check("m_total iç tutarlılık", mass["m_total"], m_total_json_sum, tol_pct=0.001)

# =============================================================================
# BÖLÜM 5: YÜK HESAPLARI (Uniform Distribution)
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 5: YÜK DOĞRULAMASI")
print("=" * 110)

# For uniform distribution (default):
L_total = n_load * W_0   # Total lift force [N]
L_half = L_total / 2     # Half-wing lift [N]

# w(y) = L_half / L_span (uniform)
w_uniform = L_half / L_span

# V_root = L_half (integral of w from 0 to L_span)
V_root = L_half

# M_root = integral of V from 0 to L_span
# For uniform: V(y) = w * (L_span - y), so M(y) = w * (L_span - y)² / 2
# M_root = w * L_span² / 2 = L_half * L_span / 2
M_root_analytical = L_half * L_span / 2

# Numerical (trapezoidal) - same as code does
N_pts = N_Rib + 1
y_arr = np.linspace(0, L_span, N_pts)
w_arr = np.full(N_pts, w_uniform)

# Shear: V(y) = integral from y to L_span of w
V_arr = np.zeros(N_pts)
for i in range(N_pts - 2, -1, -1):
    dy = y_arr[i+1] - y_arr[i]
    V_arr[i] = V_arr[i+1] + 0.5 * (w_arr[i] + w_arr[i+1]) * dy

# Bending: M(y) = integral from y to L_span of V
M_arr = np.zeros(N_pts)
for i in range(N_pts - 2, -1, -1):
    dy = y_arr[i+1] - y_arr[i]
    M_arr[i] = M_arr[i+1] + 0.5 * (V_arr[i] + V_arr[i+1]) * dy

print(f"  w_uniform = {w_uniform:.4f} N/m")
print(f"  V_root (analitik) = {V_root:.4f} N")
print(f"  V_root (sayısal) = {V_arr[0]:.4f} N")
print(f"  M_root (analitik) = {M_root_analytical:.4f} N·m")
print(f"  M_root (sayısal, {N_pts} nokta) = {M_arr[0]:.4f} N·m")

check("Fz_N (V_root)", reactions["Fz_N"], V_arr[0], tol_pct=0.1)
check("Mx_Nm (M_root)", reactions["Mx_Nm"], M_arr[0], tol_pct=1.0)

# Torsion at root
# T(y) = integral from y to L_span of (w*e + m_pitch)
# e = x_ac - x_sc, x_sc = (x_FS + x_RS) / 2
# For each station: x_ac = 0.25 * chord, x_FS = X_FS_pct/100 * chord, x_RS = X_RS_pct/100 * chord
# x_sc = chord * (X_FS_pct + X_RS_pct) / 200
# e = chord * (0.25 - (X_FS_pct + X_RS_pct) / 200)

# Pitching moment: m_pitch = M_pitch_half / L_span (uniform)
q_inf = 0.5 * rho_air * V_c**2
M_pitch_total = C_m * q_inf * S_ref * c_MGC
M_pitch_half = M_pitch_total / 2
m_pitch_uniform = M_pitch_half / L_span

chord_arr = np.array([C_r * (1 - (1 - lam) * yi / L_span) for yi in y_arr])
x_ac_arr = (x_ac_pct / 100) * chord_arr
x_FS_arr = (X_FS_pct / 100) * chord_arr
x_RS_arr = (X_RS_pct / 100) * chord_arr
x_sc_arr = (x_FS_arr + x_RS_arr) / 2
e_arr = x_ac_arr - x_sc_arr

# Torsion intensity: t(y) = w(y) * e(y) + m_pitch(y)
t_arr = w_arr * e_arr + m_pitch_uniform

# T(y) = integral from y to L_span of t
T_arr = np.zeros(N_pts)
for i in range(N_pts - 2, -1, -1):
    dy = y_arr[i+1] - y_arr[i]
    T_arr[i] = T_arr[i+1] + 0.5 * (t_arr[i] + t_arr[i+1]) * dy

check("My_Nm (T_root)", reactions["My_Nm"], T_arr[0], tol_pct=1.0)

print(f"\n  M_pitch_half = {M_pitch_half:.6f} N·m")
print(f"  e_root = {e_arr[0]*1000:.4f} mm (x_ac - x_sc at root)")
print(f"  T_root = {T_arr[0]:.6f} N·m")

# =============================================================================
# BÖLÜM 6: GERİLME DOĞRULAMASI
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 6: GERİLME DOĞRULAMASI (Root Station)")
print("=" * 110)

# At root (y=0):
M_root = M_arr[0]
T_root = T_arr[0]

# --- Skin shear stress (Bredt-Batho) ---
# A_m = w_box * h_box (rectangular enclosed area)
A_m_root = (X_RS_at_root - X_FS_at_root) * h_box_root
q_flow = T_root / (2 * A_m_root)   # N/m
tau_skin = q_flow / t_skin          # Pa

check("tau_skin_max_MPa", stress["tau_skin_max_MPa"], abs(tau_skin) / 1e6, tol_pct=1.0)

# --- Spar bending stress (box beam model) ---
# sigma_b = M * c_spar / I_box  (I_box = transformed section)
sigma_b_FS = abs(M_root) * c_FS / I_box
sigma_b_RS = abs(M_root) * c_RS / I_box

check("sigma_b_FS_max_MPa", stress["sigma_b_FS_max_MPa"], sigma_b_FS / 1e6, tol_pct=1.0)
check("sigma_b_RS_max_MPa", stress["sigma_b_RS_max_MPa"], sigma_b_RS / 1e6, tol_pct=1.0)

# --- Torsional shear stress ---
# tau_FS = |T * eta_FS| * c_FS / J_FS
# tau_RS = |T * eta_RS| * c_RS / J_RS
tau_FS = abs(T_root * eta_FS) * c_FS / J_FS
tau_RS = abs(T_root * eta_RS) * c_RS / J_RS

check("tau_FS_max_MPa", stress["tau_FS_max_MPa"], tau_FS / 1e6, tol_pct=1.0)
check("tau_RS_max_MPa", stress["tau_RS_max_MPa"], tau_RS / 1e6, tol_pct=1.0)

# --- Von Mises stress ---
sigma_vm_FS = math.sqrt(sigma_b_FS**2 + 3 * tau_FS**2)
sigma_vm_RS = math.sqrt(sigma_b_RS**2 + 3 * tau_RS**2)

check("sigma_vm_FS_max_MPa", stress["sigma_vm_FS_max_MPa"], sigma_vm_FS / 1e6, tol_pct=1.0)
check("sigma_vm_RS_max_MPa", stress["sigma_vm_RS_max_MPa"], sigma_vm_RS / 1e6, tol_pct=1.0)

# --- Allowable stresses ---
sigma_allow = sigma_u_spar / SF
tau_allow_skin = tau_u_skin / SF
tau_allow_spar = tau_u_spar / SF

check("sigma_FS_allow_MPa", stress["sigma_FS_allow_MPa"], sigma_allow / 1e6)
check("tau_skin_allow_MPa", stress["tau_skin_allow_MPa"], tau_allow_skin / 1e6)

# --- Margins ---
margin_skin = (tau_allow_skin - abs(tau_skin)) / tau_allow_skin * 100
margin_FS = (sigma_allow - sigma_vm_FS) / sigma_allow * 100
margin_RS = (sigma_allow - sigma_vm_RS) / sigma_allow * 100

check("tau_skin_margin_%", stress["tau_skin_margin_percent"], margin_skin, tol_pct=1.0)
check("sigma_FS_margin_%", stress["sigma_FS_margin_percent"], margin_FS, tol_pct=1.0)
check("sigma_RS_margin_%", stress["sigma_RS_margin_percent"], margin_RS, tol_pct=1.0)

# =============================================================================
# BÖLÜM 7: BURKULMA DOĞRULAMASI
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 7: BURKULMA (BUCKLING) DOĞRULAMASI")
print("=" * 110)

rib_spacing = L_span / N_Rib  # [m]

# tau_cr = k_s * π² * E / (12*(1-ν²)) * (t/s)²
k_s = 5.34
tau_cr = k_s * math.pi**2 * E_skin / (12 * (1 - nu_skin**2)) * (t_skin / rib_spacing)**2

# sigma_cr = k_c * π² * E / (12*(1-ν²)) * (t/s)²
k_c = 4.0
sigma_cr = k_c * math.pi**2 * E_skin / (12 * (1 - nu_skin**2)) * (t_skin / rib_spacing)**2

check("tau_cr_min_MPa", buckling["tau_cr_min_MPa"], tau_cr / 1e6, tol_pct=1.0)
check("sigma_cr_min_MPa", buckling["sigma_cr_min_MPa"], sigma_cr / 1e6, tol_pct=1.0)

# Skin compression stress at root
# sigma_skin_comp = n_ratio * |M| * (h_box/2) / I_box
sigma_skin_comp = n_ratio * abs(M_root) * (h_box_root / 2) / I_box

check("sigma_skin_comp_max_MPa", buckling["sigma_skin_comp_max_MPa"], sigma_skin_comp / 1e6, tol_pct=1.0)

# R_combined = (tau/tau_cr)² + (sigma/sigma_cr)
R_combined = (abs(tau_skin) / tau_cr)**2 + abs(sigma_skin_comp) / sigma_cr

check("R_combined_max", buckling["R_combined_max"], R_combined, tol_pct=1.0)

# =============================================================================
# BÖLÜM 8: KABUL KRİTERLERİ DOĞRULAMASI
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 8: KABUL KRİTERLERİ (Acceptance Criteria)")
print("=" * 110)

criteria = [
    ("1. tau_skin < tau_allow",
     abs(tau_skin) < tau_allow_skin,
     f"{abs(tau_skin)/1e6:.4f} < {tau_allow_skin/1e6:.1f} MPa"),

    ("2. FS interaction < 1.0",
     (sigma_b_FS / sigma_allow) + (tau_FS / tau_allow_spar)**2 < 1.0,
     f"R = {(sigma_b_FS/sigma_allow) + (tau_FS/tau_allow_spar)**2:.4f}"),

    ("3. RS interaction < 1.0",
     (sigma_b_RS / sigma_allow) + (tau_RS / tau_allow_spar)**2 < 1.0,
     f"R = {(sigma_b_RS/sigma_allow) + (tau_RS/tau_allow_spar)**2:.4f}"),

    ("4. A_Act_FS > A_Cri_FS",
     A_FS > A_Cri_FS,
     f"{A_FS*1e6:.2f} > {A_Cri_FS*1e6:.2f} mm²"),

    ("5. A_Act_RS > A_Cri_RS",
     A_RS > A_Cri_RS,
     f"{A_RS*1e6:.2f} > {A_Cri_RS*1e6:.2f} mm²"),

    ("6. d_FS + 10 < h_box",
     d_FS_mm + 10 < h_box_root * 1000,
     f"{d_FS_mm}+10 = {d_FS_mm+10} < {h_box_root*1000:.2f} mm"),

    ("7. d_RS + 10 < h_box",
     d_RS_mm + 10 < h_box_root * 1000,
     f"{d_RS_mm}+10 = {d_RS_mm+10} < {h_box_root*1000:.2f} mm"),

    ("8. Shear buckling (tau < tau_cr)",
     abs(tau_skin) < tau_cr,
     f"{abs(tau_skin)/1e6:.4f} < {tau_cr/1e6:.4f} MPa"),
]

for name, passed, detail in criteria:
    status = "PASS" if passed else "FAIL"
    marker = "  ✓" if passed else "  ✗"
    print(f"{marker} {name:40s} | {detail:50s} | {status}")

# =============================================================================
# BÖLÜM 9: TİP TWIST (yaklaşık)
# =============================================================================

print("\n" + "=" * 110)
print("BÖLÜM 9: TIP TWIST DOĞRULAMASI")
print("=" * 110)

# Twist rate: dθ/dz = q / (2 * A_m * G * t) * perimeter
# For single cell: perimeter ≈ 2*(w_box + h_box)
# θ_tip = integral of twist_rate from 0 to L_span

# Compute at each station
theta_arr = np.zeros(N_pts)
for i in range(N_pts):
    yi = y_arr[i]
    chord_i = chord_arr[i]
    h_i = t_over_c * chord_i
    x_FS_i = (X_FS_pct / 100) * chord_i
    x_RS_i = (X_RS_pct / 100) * chord_i
    w_i = x_RS_i - x_FS_i
    A_m_i = w_i * h_i
    perimeter_i = 2 * (w_i + h_i)

    if A_m_i > 0:
        q_i = T_arr[i] / (2 * A_m_i)
        twist_rate_i = (q_i * perimeter_i) / (2 * A_m_i * G_skin * t_skin)
    else:
        twist_rate_i = 0.0
    theta_arr[i] = twist_rate_i

# Integrate twist rate from root to tip: θ(y) = ∫₀ʸ (dθ/dz) dz
theta_tip_rad = np.trapz(theta_arr, y_arr)
theta_tip_deg = abs(math.degrees(theta_tip_rad))  # abs - yön önemli değil

check("theta_tip_deg", stress["theta_tip_deg"], theta_tip_deg, tol_pct=2.0,
      unit="(sayısal integrasyon farkı olabilir)")

# =============================================================================
# ÖZET
# =============================================================================

print("\n" + "=" * 110)
print("DOĞRULAMA ÖZETİ")
print("=" * 110)
total = pass_count + fail_count + warn_count
print(f"  Toplam kontrol : {total}")
print(f"  PASS           : {pass_count}")
print(f"  WARN           : {warn_count}")
print(f"  FAIL           : {fail_count}")
print(f"  Başarı oranı   : {pass_count/total*100:.1f}%")

if fail_count == 0:
    print(f"\n  *** TÜM KONTROLLER GEÇTİ ***")
else:
    print(f"\n  !!! {fail_count} KONTROL BAŞARISIZ - DETAYLARI YUKARDA İNCELEYİN !!!")

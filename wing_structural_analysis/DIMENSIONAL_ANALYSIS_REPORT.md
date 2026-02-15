# Boyut Analizi Raporu - Wing Structural Analysis

**Tarih:** 2026-02-15
**Hazirlayan:** Codex-Consultant Agent
**Yontem:** Her formul icin giris degiskenleri birim analizi + Codex CLI dogrulama

---

## OZET

| Modul | Formul Sayisi | Dogru | Hata | Uyari |
|-------|:---:|:---:|:---:|:---:|
| buckling.py | 6 | 6 | 0 | 0 |
| spars.py | 12 | 11 | 0 | 1 |
| loads.py | 8 | 8 | 0 | 0 |
| torsion.py | 5 | 5 | 0 | 0 |
| geometry.py | 8 | 8 | 0 | 0 |
| ribs.py | 4 | 4 | 0 | 0 |
| optimization.py | 3 | 1 | 1 | 1 |
| materials.py | 2 | 2 | 0 | 0 |
| **TOPLAM** | **48** | **45** | **1** | **2** |

---

## 1. buckling.py - TUM FORMULLER DOGRU

### 1.1 skin_shear_buckling_critical (satir 42)
```
tau_cr = k_s * pi^2 * E / (12*(1-nu^2)) * (t/s)^2
```
- k_s [-] * pi^2 [-] * E [Pa] / (12*(1-nu^2)) [-] * (t [m] / s [m])^2 [-]
- = E [Pa] * [-] = **[Pa]** -- DOGRU

### 1.2 skin_compression_buckling_critical (satir 64)
```
sigma_cr = k_c * pi^2 * E / (12*(1-nu^2)) * (t/s)^2
```
- Ayni yapida, sonuc **[Pa]** -- DOGRU

### 1.3 rib_web_shear_buckling_critical (satir 86)
```
tau_cr_rib = k_s * pi^2 * E / (12*(1-nu^2)) * (t_rib/h_box)^2
```
- Ayni yapida, sonuc **[Pa]** -- DOGRU

### 1.4 combined_interaction_ratio (satir 111-116)
```
R = (tau/tau_cr)^2 + sigma/sigma_cr
```
- (Pa/Pa)^2 + Pa/Pa = [-] + [-] = **[-]** -- DOGRU

### 1.5 compute_s_max_shear (satir 142)
```
s_max = t * sqrt(k_s * pi^2 * E / (12*(1-nu^2) * tau))
```
- t [m] * sqrt([-] * E [Pa] / ([-] * tau [Pa])) = t [m] * sqrt([-]) = **[m]** -- DOGRU

### 1.6 compute_s_max_compression (satir 164)
```
s_max = t * sqrt(k_c * pi^2 * E / (12*(1-nu^2) * sigma))
```
- Ayni yapida, sonuc **[m]** -- DOGRU

---

## 2. spars.py - 11/12 DOGRU, 1 UYARI

### 2.1 SparProperties.area (satir 35)
```
A = (pi/4) * (d_outer^2 - d_inner^2)
```
- (pi/4) [-] * ([m]^2 - [m]^2) = **[m^2]** -- DOGRU

### 2.2 SparProperties.I (satir 40)
```
I = (pi/64) * (d_outer^4 - d_inner^4)
```
- (pi/64) [-] * ([m]^4 - [m]^4) = **[m^4]** -- DOGRU

### 2.3 SparProperties.J (satir 51)
```
J = (pi/32) * (d_outer^4 - d_inner^4) = 2*I
```
- **[m^4]** -- DOGRU (J = 2I icin ince cidali tup dogru)

### 2.4 bending_stress (satir 180)
```
sigma_b = M * c / I
```
- [N*m] * [m] / [m^4] = [N/m^2] = **[Pa]** -- DOGRU

### 2.5 torsional_shear_stress (satir 199)
```
tau = T * r / J
```
- [N*m] * [m] / [m^4] = [N/m^2] = **[Pa]** -- DOGRU

### 2.6 von_mises_stress (satir 215)
```
sigma_vm = sqrt(sigma^2 + 3*tau^2)
```
- sqrt([Pa]^2 + 3*[Pa]^2) = **[Pa]** -- DOGRU

### 2.7 critical_area_bending (satir 427)
```
A_crit = shape_factor * 2 * |M| / (sigma_allow * c_dist)
```
- [-] * 2 * [N*m] / ([Pa] * [m]) = [N*m] / ([N/m^2] * [m]) = [N*m] / [N/m] = **[m^2]** -- DOGRU
- Turetim: Ince cidali tup icin I/A ~ c^2/2, sigma = Mc/I = 2M/(Ac), dolayisiyla A = 2M/(sigma*c). Bu dogru.

### 2.8 tip_deflection_uniform_load (satir 449)
```
delta = w * L^4 / (8 * E * I)
```
- [N/m] * [m]^4 / ([Pa] * [m^4]) = [N*m^3] / ([N/m^2] * [m^4]) = [N*m^3] / [N*m^2] = **[m]** -- DOGRU

### 2.9 tip_deflection_point_load (satir 467)
```
delta = P * L^3 / (3 * E * I)
```
- [N] * [m]^3 / ([Pa] * [m^4]) = [N*m^3] / [N*m^2] = **[m]** -- DOGRU

### 2.10 tip_deflection_from_moment (satir 493-496)
```
integrand = (L_span - y) * M / (E * I)
delta_tip = trapz(integrand, y)
```
- integrand: [m] * [N*m] / ([Pa] * [m^4]) = [m * N*m] / [N*m^2] = [-]
- integral: [-] * dy [m] = **[m]** -- DOGRU

### 2.11 compute_box_beam_section (satir 560-564)
```
I_skin_parallel = 2 * w_box * t_skin * d^2     -- Steiner terimi
I_skin_self = 2 * w_box * t_skin^3 / 12
I_skin_transformed = n * (I_skin_parallel + I_skin_self)
I_total = I_FS + I_RS + I_skin_transformed
```
- I_skin_parallel: 2 * [m] * [m] * [m]^2 = **[m^4]** -- DOGRU
- I_skin_self: 2 * [m] * [m]^3 / 12 = **[m^4]** -- DOGRU
- n = E_skin/E_spar [-], I_skin_transformed: [-] * [m^4] = **[m^4]** -- DOGRU
- I_total: [m^4] + [m^4] + [m^4] = **[m^4]** -- DOGRU

### 2.12 analyze_box_beam_stress (satir 621-631) -- UYARI
```
sigma_b_FS = |M| * c_FS / I_total
sigma_skin = n_skin * |M| * (h_box/2) / I_total
tau_FS_val = |T * eta_FS| * c_FS / J_FS
```
- sigma_b: [N*m] * [m] / [m^4] = **[Pa]** -- DOGRU
- sigma_skin: [-] * [N*m] * [m] / [m^4] = **[Pa]** -- DOGRU
- tau_FS: [N*m] * [-] * [m] / [m^4] = **[Pa]** -- DOGRU

**UYARI:** Torsion paylasimi (eta_FS/eta_RS) pozisyon bazli yapiliyor ama torsion direnci pozisyon bazli degil, J bazli olmali. Ancak bu bir boyut hatasi degil, fiziksel model secimi.

### 2.13 spar_mass (satir 394)
```
m = A * L * rho
```
- [m^2] * [m] * [kg/m^3] = **[kg]** -- DOGRU

---

## 3. loads.py - TUM FORMULLER DOGRU

### 3.1 L_total (satir 41)
```
L_total = n * W0
```
- [-] * [N] = **[N]** -- DOGRU

### 3.2 q_inf (satir 52)
```
q_inf = 0.5 * rho * V_c^2
```
- 0.5 * [kg/m^3] * [m/s]^2 = [kg/(m*s^2)] = **[Pa]** -- DOGRU

### 3.3 M_pitch_total (satir 57)
```
M_pitch = C_m * q_inf * S_ref * c_MGC
```
- [-] * [Pa] * [m^2] * [m] = [N/m^2] * [m^3] = **[N*m]** -- DOGRU

### 3.4 lift_distribution_uniform (satir 83)
```
w = L_half / L_span
```
- [N] / [m] = **[N/m]** -- DOGRU

### 3.5 lift_distribution_elliptic (satir 101)
```
w0 = 4 * L_half / (pi * L_span)
w = w0 * sqrt(1 - eta^2)
```
- w0: [N] / [m] = **[N/m]** -- DOGRU

### 3.6 compute_shear_force (satir 140-150)
```
V(y) = integral from y to L of w(xi) dxi
```
- [N/m] * [m] = **[N]** -- DOGRU (trapezoidal integration)

### 3.7 compute_bending_moment (satir 166-172)
```
M(y) = integral from y to L of V(xi) dxi
```
- [N] * [m] = **[N*m]** -- DOGRU

### 3.8 compute_torsion_intensity (satir 296-297)
```
t_L = w * e
return t_L + m_pitch
```
- w [N/m] * e [m] = [N*m/m] = [N] ... HAYIR: w*e = [N/m]*[m] = [N] ...

Biraz, burada t [N*m/m] olmali (torsion intensity). w [N/m] * e [m] = [N] mu, yoksa [N*m/m] mi?

Aslinda: w [N/m] (kuvvet/birim uzunluk), e [m] (kol), carpim: [N/m] * [m] = [N]. Ama torsion intensity "torque per unit span" olmali, yani [N*m/m].

Tekrar dusunelim: w(y) [N/m], e(y) [m].
t_L(y) = w(y) * e(y) = [N/m] * [m] = [N]. Ama beklenen birim [N*m/m].

[N] = [N*m/m] MI? Evet! [N] = [N*m/m]. Cunku [N*m/m] = [N].

**DOGRU.** [N/m]*[m] = [N] = [N*m/m].

m_pitch [N*m/m] ile toplam: [N*m/m] + [N*m/m] = **[N*m/m]** -- DOGRU

### 3.9 compute_torsion (satir 313-318)
```
T(y) = integral from y to L of t(xi) dxi
```
- [N*m/m] * [m] = **[N*m]** -- DOGRU

---

## 4. torsion.py - TUM FORMULLER DOGRU

### 4.1 bredt_batho_single_cell (satir 45)
```
q = T / (2 * A_m)
```
- [N*m] / (2 * [m^2]) = [N/m] ... Hmm, [N*m]/[m^2] = **[N/m]**. -- DOGRU (shear flow)

### 4.2 shear_stress_from_flow (satir 63)
```
tau = q / t
```
- [N/m] / [m] = [N/m^2] = **[Pa]** -- DOGRU

### 4.3 twist_rate_single_cell (satir 86-87)
```
delta = perimeter / t_avg    -- [m] / [m] = [1/m]...
```
Dur. perimeter [m], t_avg [m]. delta = P/t = [m]/[m] = [-]? Hayir:
delta = integral(ds/t) ve burada ds [m], t [m], yani ds/t = [-], tum integral de [-].

Ama kodda `delta = perimeter / t_avg`, bu [m]/[m] = [-]. Sonra:
```
twist_rate = (q * delta) / (2 * A_m * G)
```
- [N/m] * [-] / (2 * [m^2] * [Pa]) = [N/m] / ([m^2] * [N/m^2]) = [N/m] / [N] = **[1/m]** = **[rad/m]** -- DOGRU

### 4.4 solve_multicell_torsion (satir 151-232)
- Torque equilibrium: T = sum(2*A_R*q_R), [N*m] = [m^2]*[N/m] = [N*m] -- DOGRU
- Compatibility: twist rate = (1/(2*A*G)) * (q*delta - ...), birimler tutarli -- DOGRU

### 4.5 analyze_all_stations (satir 303-338)
- perimeter = 2*(w + h): [m] + [m] = [m] -- DOGRU
- Tum fonksiyon cagrilari dogru parametrelerle -- DOGRU

---

## 5. geometry.py - TUM FORMULLER DOGRU

### 5.1 PlanformParams.from_input (satir 59-68)
```
C_r = C_r_mm / 1000    -- [mm] / 1000 = [m] -- DOGRU
Y_bar = Y_bar_mm / 1000  -- [mm] / 1000 = [m] -- DOGRU
```

### 5.2 chord_at_station (satir 86)
```
c(y) = C_r - (C_r - C_tip) * (y / L_span)
```
- [m] - [m] * ([m]/[m]) = [m] - [m] = **[m]** -- DOGRU

### 5.3 compute_wing_box_section (satir 197-198)
```
h_box = t_c * chord       -- [-] * [m] = [m] -- DOGRU
A_m = (x_RS - x_FS) * h_box  -- [m] * [m] = [m^2] -- DOGRU
```

### 5.4 compute_spar_sweep_angle (satir 256-259)
```
term2 = (4 * (x_ac% - X_spar%) * (1-lambda)) / (x_ac% * AR * (1+lambda))
```
- Tum terimler [%] ve [-], boyutsuz oran -- DOGRU (sonuc [derece])

### 5.5 arc_length_LE_to_spar (satir 301-308)
```
term1 = sqrt((h/2)^2 + (2*X)^2)
term2 = ((h/2)^2 / (4*lam*X)) * asinh(4*X/h)
```
- term1: sqrt([m]^2 + [m]^2) = [m]
- term2: [m]^2 / ([-]*[m]) * [-] = [m^2]/[m] = [m]
- Toplam: **[m]** -- DOGRU

### 5.6 compute_skin_area_half_wing (satir 338-339)
```
S_skin_half = 2 * avg_chord * L_span
```
- 2 * [m] * [m] = **[m^2]** -- DOGRU

### 5.7 compute_rib_areas_root (satir 363-366)
```
S_Rib_LE_FS = (2/3) * x_FS_root * h_box_root
```
- (2/3) * [m] * [m] = **[m^2]** -- DOGRU

### 5.8 SparGeometry.from_mm (satir 159-162)
```
d_outer = d_outer_mm / 1000  -- [mm]/1000 = [m] -- DOGRU
t_wall = t_wall_mm / 1000    -- [mm]/1000 = [m] -- DOGRU
```

---

## 6. ribs.py - TUM FORMULLER DOGRU

### 6.1 RibGeometry.S_LE_FS (satir 59-61)
```
Parabolic: (2/3) * x_FS * h_box = [m]*[m] = [m^2] -- DOGRU
Rectangular: h_box * x_FS = [m]*[m] = [m^2] -- DOGRU
```

### 6.2 RibGeometry.mass (satir 89)
```
m = S_total * t_rib * density
```
- [m^2] * [m] * [kg/m^3] = **[kg]** -- DOGRU

### 6.3 compute_rib_spacing (satir 107)
```
spacing = L_span / N_Rib
```
- [m] / [-] = **[m]** -- DOGRU

### 6.4 adaptive_rib_insertion: s_target hesabi (satir 446)
```
s_target = min(s_max_shear, s_max_comp) * margin
```
- min([m], [m]) * [-] = **[m]** -- DOGRU

---

## 7. optimization.py - 1 HATA, 1 UYARI

### 7.1 t_skin donusumu (satir 298)
```
t_skin = self.opt_config.t_skin_mm / 1000
```
- [mm] / 1000 = **[m]** -- DOGRU

### 7.2 Spar length hesabi (satir 446-447)
```
L_FS = planform.L_span / cos(radians(Lambda_FS))
```
- [m] / [-] = **[m]** -- DOGRU

### 7.3 *** HATA *** A_Cri formulu (satir 394-395)
```python
result.A_Cri_FS = (n * W_0 * Y_bar * 0.5 * result.eta_FS) / (2 * sigma_max_spar * c_FS)
result.A_Cri_RS = (n * W_0 * Y_bar * 0.5 * result.eta_RS) / (2 * sigma_max_spar * c_RS)
```

**Boyut analizi:**
- Pay: n [-] * W_0 [N] * Y_bar [m] * 0.5 [-] * eta [-] = [N*m]
- Payda: 2 [-] * sigma_max [Pa] * c [m] = [N/m^2] * [m] = [N/m]
- Sonuc: [N*m] / [N/m] = **[m^2]** -- Boyutsal olarak DOGRU

**Sayi faktor HATASI:**
- Bu formula A_Cri = n*W0*Y_bar*eta / (4*sigma*c)
- spars.py (satir 427): A_crit = 2*M/(sigma*c) ile M_spar = eta * n*W0*Y_bar/2 kullanilinca:
  A_crit = 2*(eta*n*W0*Y_bar/2) / (sigma*c) = eta*n*W0*Y_bar / (sigma*c)
- ORAN: optimization.py / spars.py = 1/4

**optimization.py, gerekli alanin 4 katini KUCUK hesapliyor.**

Bu, yapisal olarak yetersiz spar gecmesine izin verebilir. Kontrol `A_Act > A_Cri` seklinde yapildigi icin, A_Cri kucuk hesaplaninca kontrol kolayca gecer.

**ONERILEN DUZELTME:**
```python
# DOGRU: M_root_approx = n * W_0 * Y_bar / 2 (half-wing)
# M_spar = eta * M_root
# A_cri = 2 * M_spar / (sigma * c) = eta * n * W_0 * Y_bar / (sigma * c)
M_root_approx = n * W_0 * Y_bar / 2
result.A_Cri_FS = 2 * (result.eta_FS * M_root_approx) / (sigma_max_spar * c_FS)
result.A_Cri_RS = 2 * (result.eta_RS * M_root_approx) / (sigma_max_spar * c_RS)
```

**NOT:** Ayni hata main.py satir 516-517'de de mevcut.

**ONEMLI NUANS:** CLAUDE.md referans dokumani da ayni "hatali" formulunu listeliyor:
```
A_Cri_FS = (n * W_0 * Y_bar * 0.5 * eta_FS) / (2 * sigma_max * c_FS)
```
Bu, ya referansin kendisi hatali, ya da farkli bir I/A approximation'i kullaniliyor. Ince cidali tup icin spars.py turetimi (I/A ~ c^2/2) standart yaklasimdir.

### 7.4 UYARI: sigma_max_spar vs sigma_allow_spar
```python
sigma_max_spar = mat_spar.sigma_u     # sigma_u kullaniliyor (SF uygulanmamis)
```
Oysa spar stress interaction check (satir 398-409) SF ile bolunmus allowable kullaniyor. A_Cri formulunde sigma_u (SF'siz) kullanmak, kritik alanindan daha kucuk cikmasina neden olur. Bu bilinçli bir secim olabilir (ultimate ile sizing) ama tutarsizlik var.

---

## 8. materials.py - TUM FORMULLER DOGRU

### 8.1 Shear modulus (satir 26)
```
G = E / (2 * (1 + nu))
```
- [Pa] / (2 * (1 + [-])) = [Pa] / [-] = **[Pa]** -- DOGRU

### 8.2 get_allowables (satir 39-40)
```
sigma_allow = sigma_u / SF
tau_allow = tau_u / SF
```
- [Pa] / [-] = **[Pa]** -- DOGRU

---

## MM -> M DONUSUM KONTROLU

Tum dosyalarda mm->m donusumleri dogrulandirdi:

| Dosya | Satir | Donusum | Durum |
|-------|-------|---------|-------|
| geometry.py | 65 | C_r_mm / 1000 | DOGRU |
| geometry.py | 67 | Y_bar_mm / 1000 | DOGRU |
| geometry.py | 159-162 | d_outer_mm / 1000, t_wall_mm / 1000 | DOGRU |
| spars.py | 90-91 | d_outer_mm / 1000, t_wall_mm / 1000 | DOGRU |
| ribs.py | 39 | t_rib_mm / 1000 | DOGRU |
| optimization.py | 298 | t_skin_mm / 1000 | DOGRU |
| optimization.py | 475 | t_rib_mm / 1000 | DOGRU |
| optimization.py | 480 | s_min_mm / 1000 | DOGRU |
| main.py | 516 | Y_bar_mm / 1000 | DOGRU |

---

## MPA -> PA DONUSUM KONTROLU

Malzeme degerleri materials.py'de zaten Pa cinsinden depolaniyor (orn. 572e6 Pa).
Tum hesaplamalar SI birimlerinde yapiliyor, cikis formatlamada Pa->MPa donusumu `/1e6` ile yapiliyor.
**Hata yok.**

---

## SONUC VE ONERILER

### Bulunan Hatalar

**1. optimization.py satir 394-395 ve main.py satir 516-517: A_Cri formulu faktor hatasi**
- Mevcut formula, ince cidali tup icin gerekli minimum alani 4 kat kucuk hesapliyor
- Bu, yetersiz spar boyutlarinin kabul edilmesine yol acabilir
- Pratikte optimization.py'deki diger check'ler (interaction formula) bu konfiigurasyonlari yakalayabilir, ancak yalnizca A_Cri check'ine guvenilemez

### Uyarilar

**1. Torsion load sharing (spars.py satir 629-630)**
- Torsion pozisyon bazli paylastiriliyor (eta_FS, eta_RS)
- Fiziksel olarak torsion J bazli paylasilmali
- Boyut hatasi degil, fiziksel model kararı

**2. sigma_u vs sigma_u/SF tutarsizligi (optimization.py satir 390)**
- A_Cri'de sigma_u (SF'siz) kullaniliyor
- Diger kontrollerde sigma_u/SF kullaniliyor
- Tasarim felsefesi olarak tutarsiz

### Her Sey Dogru Olan Moduller
- buckling.py: Tum 6 formul boyutsal olarak tutarli
- loads.py: Tum 8 formul boyutsal olarak tutarli
- torsion.py: Tum 5 formul boyutsal olarak tutarli (Bredt-Batho, twist rate dahil)
- geometry.py: Tum 8 formul ve donusum boyutsal olarak tutarli
- ribs.py: Tum 4 formul boyutsal olarak tutarli
- materials.py: Tum 2 formul boyutsal olarak tutarli

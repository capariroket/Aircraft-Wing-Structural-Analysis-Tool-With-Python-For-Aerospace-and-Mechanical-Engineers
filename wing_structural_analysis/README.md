# Half-Wing Structural Sizing Tool

**Yarı-Kanat Yapısal Boyutlandırma ve Optimizasyon Aracı**

Bu araç, uçak kanat yapılarının ön tasarım aşamasında kullanılmak üzere geliştirilmiş bir Python tabanlı hesaplama ve optimizasyon paketidir. Grid search yöntemiyle en hafif yapısal konfigürasyonu bulur.

---

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Modül Yapısı](#modül-yapısı)
- [Teorik Arka Plan](#teorik-arka-plan)
- [Input Parametreleri](#input-parametreleri)
- [Output Değerleri](#output-değerleri)
- [Formüller](#formüller)
- [Kabuller ve Sınırlamalar](#kabuller-ve-sınırlamalar)
- [Örnek Kullanım](#örnek-kullanım)

---

## ✨ Özellikler

- **Grid Search Optimizasyonu**: Kullanıcı tanımlı aralıklarda tüm kombinasyonları tarar
- **Çoklu Yük Dağılımı**: Uniform ve elliptik lift dağılımı desteği
- **Bredt-Batho Torsion**: Kapalı kesit shear flow hesabı
- **Von Mises Stress**: Spar'larda kombine stress analizi
- **Otomatik Malzeme Seçimi**: Dahili malzeme veritabanı
- **Görselleştirme**: Matplotlib ile otomatik grafik üretimi
- **JSON Export**: Sonuçların programatik kullanımı için

---

## 🚀 Kurulum

### Gereksinimler

```bash
Python >= 3.8
NumPy
Matplotlib
```

### Kurulum

```bash
git clone https://github.com/capariroket/wing_structural_analysis.git
cd wing_structural_analysis
pip install numpy matplotlib
```

---

## ⚡ Hızlı Başlangıç

```bash
python main.py
```

Program interaktif olarak parametreleri sorar. Her soruda:
- **Değer girmek için**: Sayıyı yazıp ENTER
- **Default kullanmak için**: Direkt ENTER

**Önemli**: Ondalık ayracı olarak **nokta (.)** kullanın: `0.12`, `1.5`

---

## 📁 Modül Yapısı

```
wing_structural_analysis/
│
├── main.py              # Ana program - interaktif CLI
├── materials.py         # Malzeme veritabanı ve seçimi
├── geometry.py          # Planform ve wing-box geometrisi
├── loads.py             # Yük dağılımı (w, V, M, T)
├── torsion.py           # Bredt-Batho shear flow
├── spars.py             # Spar stress ve inertia
├── ribs.py              # Rib geometri ve kütle
├── optimization.py      # Grid search algoritması
├── plots.py             # Grafik üretimi
├── reporting.py         # Sonuç raporları
└── __init__.py          # Paket tanımı
```

### Modül Detayları

#### `materials.py`
Malzeme veritabanını yönetir. Dahili malzemeler:

| ID | Malzeme | E [GPa] | σ_u [MPa] | ρ [kg/m³] |
|----|---------|---------|-----------|-----------|
| 1 | AL7075-T6 | 71.7 | 572 | 2810 |
| 2 | AL2024-T3 | 73.1 | 483 | 2780 |
| 3 | CFRP_UD | 135.0 | 1500 | 1600 |
| 4 | GFRP | 40.0 | 600 | 1900 |
| 5 | PLA | 3.5 | 50 | 1250 |
| 6 | STEEL_4130 | 205.0 | 670 | 7850 |

Allowable stress hesabı:
```
σ_allow = σ_u / SF
τ_allow = τ_u / SF
```

#### `geometry.py`
Planform geometrisini hesaplar:
- Chord dağılımı (lineer taper)
- Rib istasyonları
- Wing-box kesit özellikleri
- Spar sweep açıları

#### `loads.py`
Yük ve moment dağılımlarını hesaplar:
- Lift dağılımı w(y)
- Shear force V(y)
- Bending moment M(y)
- Pitching moment dağılımı
- Torsion T(y)

#### `torsion.py`
Bredt-Batho teorisi ile shear flow:
- Tek hücre analizi
- Çok hücreli analiz (opsiyonel)
- Twist rate hesabı

#### `spars.py`
Spar yapısal analizi:
- Dairesel tüp kesit özellikleri
- Bending ve shear stress
- Von Mises eşdeğer stress
- Defleksiyon hesabı

#### `ribs.py`
Rib geometri ve kütle:
- Rib alanları
- Kütle hesabı
- Spacing hesabı

#### `optimization.py`
Grid search optimizasyonu:
- Tasarım uzayı tanımı
- Geometri validasyonu
- Acceptance kriterleri
- En iyi çözüm seçimi

#### `plots.py`
Matplotlib grafikleri:
- w(y), V(y), M(y) dağılımları
- T(y), q(y), τ(y) torsion grafikleri
- σ_vm spar stress grafikleri
- Planform görünümü

#### `reporting.py`
Sonuç raporlama:
- Metin tabanlı rapor
- JSON export
- Uyarı sistemi

---

## 📐 Teorik Arka Plan

### Koordinat Sistemi

```
       y (span)
       ↑
       │
       │    ← Tip (y = L_span)
       │
       │
       └────→ x (chord)
      Root (y = 0)
```

- **x**: Chordwise (LE'den TE'ye)
- **y**: Spanwise (root'tan tip'e)
- **z**: Yukarı (lift yönü)

### Cantilever Beam Modeli

Kanat, root'ta ankastre bir kiriş olarak modellenir:
- Root (y=0): Sabit mesnet
- Tip (y=L_span): Serbest uç
- Yükler tip'ten root'a entegre edilir

---

## 📥 Input Parametreleri

### A1) Uçuş Bilimi / Aero Parametreler

| Parametre | Sembol | Birim | Açıklama |
|-----------|--------|-------|----------|
| C_m | C_m | - | Pitching moment katsayısı |
| AR | AR | - | Aspect ratio (kanat açıklığı oranı) |
| λ | lambda | - | Taper ratio (tip chord / root chord) |
| V_c | V_c | m/s | Uçuş hızı |
| n | n | - | Load factor (yük katsayısı) |
| S_ref | S_ref | m² | Kanat referans alanı |
| W_0 | W_0 | N | MTOW (maksimum kalkış ağırlığı) |
| ρ | rho | kg/m³ | Hava yoğunluğu |
| x_ac | x_ac | % | Aerodinamik merkez konumu |
| Λ_ac | Lambda_ac | deg | AC'de sweep açısı |

### A2) Geometri Parametreleri

| Parametre | Sembol | Birim | Açıklama |
|-----------|--------|-------|----------|
| b | b | m | Wingspan (toplam kanat açıklığı) |
| t/c | t_over_c | - | Kalınlık/chord oranı |
| Ȳ | Y_bar | mm | MGC'nin root'tan uzaklığı |
| c_MGC | c_MGC | m | Mean geometric chord |
| C_r | C_r | mm | Root chord uzunluğu |

### A3) Yapısal Parametreler

| Parametre | Birim | Açıklama |
|-----------|-------|----------|
| t_skin | mm | Skin kalınlığı |
| SF | - | Emniyet katsayısı (Safety Factor) |

### A4) Tasarım Aralıkları (Grid Search)

Her parametre için min/max/step tanımlanır:

**Rib:**
- N_Rib: Rib sayısı (bay sayısı)
- t_rib: Rib kalınlığı [mm]

**Front Spar:**
- X_FS%: Konum [% chord]
- d_FS_outer: Dış çap [mm]
- t_FS: Duvar kalınlığı [mm]

**Rear Spar:**
- X_RS%: Konum [% chord]
- d_RS_outer: Dış çap [mm]
- t_RS: Duvar kalınlığı [mm]

---

## 📤 Output Değerleri

### Optimal Konfigürasyon

| Output | Birim | Açıklama |
|--------|-------|----------|
| N_Rib | - | Rib sayısı |
| Λ_FS, Λ_RS | deg | Spar sweep açıları |
| η_FS, η_RS | % | Load sharing oranları |
| X_FS, X_RS | mm | Spar konumları (root'ta) |
| L_FS, L_RS | mm | Spar uzunlukları (= b/2) |
| A_(Act-FS/RS) | mm² | Gerçek spar kesit alanı |
| A_(Cri-FS/RS) | mm² | Kritik (minimum) kesit alanı |
| I_FS, I_RS | mm⁴ | Spar atalet momentleri |
| L_(Skin root LE-FS) | mm | Skin arc length (LE to FS) |
| L_(Skin root FS-RS) | mm | Skin arc length (FS to RS) |
| S_(Rib LE-FS) | mm² | Rib alanı (LE to FS) |
| S_(Rib FS-RS) | mm² | Rib alanı (FS to RS) |
| S_Rib | mm² | Toplam rib alanı |

### Kütle Tablosu

| Bileşen | Formül |
|---------|--------|
| m_skin | S_skin × t_skin × ρ_skin |
| m_FS | A_FS × L_span × ρ_spar |
| m_RS | A_RS × L_span × ρ_spar |
| m_ribs | N_Rib × S_rib × t_rib × ρ_rib |
| m_total | Σ (tüm bileşenler) |

### Stress Sonuçları

| Output | Birim | Açıklama |
|--------|-------|----------|
| τ_skin_max | MPa | Maksimum skin shear stress |
| σ_vm_FS_max | MPa | Front spar max von Mises |
| σ_vm_RS_max | MPa | Rear spar max von Mises |
| Safety Margin | % | (Allow - Actual) / Allow × 100 |

### Root Reaksiyon Vektörleri

```
Force:  [Fx, Fy, Fz] [N]
Moment: [Mx, My, Mz] [N·m]
```

Koordinat sistemi (body axes):
- x: Aft (arkaya pozitif)
- y: Starboard (sağ kanada pozitif)
- z: Up (yukarı pozitif)

---

## 📊 Formüller

### Planform Geometrisi

**Half-span:**
```
L_span = b / 2
```

**Tip chord:**
```
c_tip = λ × c_root
```

**Chord dağılımı (lineer taper):**
```
c(y) = c_root - (c_root - c_tip) × (y / L_span)
```

**Rib istasyonları:**
```
y_i = i × (L_span / N_Rib),  i = 0, 1, ..., N_Rib
```

### Wing-Box Geometrisi

**Spar konumları:**
```
x_FS(y) = (X_FS% / 100) × c(y)
x_RS(y) = (X_RS% / 100) × c(y)
```

**Box yüksekliği:**
```
h_box(y) = (t/c) × c(y)
```

**Enclosed area (tek hücre):**
```
A_m(y) = (x_RS(y) - x_FS(y)) × h_box(y)
```

### Yük Dağılımı

**Toplam lift:**
```
L_total = n × W_0
L_half = L_total / 2
```

**Uniform dağılım:**
```
w(y) = L_half / L_span
```

**Elliptic dağılım:**
```
w(y) = w_0 × √(1 - (y/L_span)²)
w_0 = 4 × L_half / (π × L_span)
```

**Shear force (tip'ten root'a):**
```
V(y) = ∫_y^L_span w(ξ) dξ
```

**Bending moment:**
```
M(y) = ∫_y^L_span V(ξ) dξ
```

### Pitching Moment

**Toplam pitching moment:**
```
q_∞ = 0.5 × ρ × V_c²
M_pitch_total = C_m × q_∞ × S_ref × c_MGC
M_pitch_half = M_pitch_total / 2
```

### Torsion

**Shear center (preliminary):**
```
x_sc(y) = (x_FS(y) + x_RS(y)) / 2
```

**Eksen kaçıklığı:**
```
e(y) = x_ac(y) - x_sc(y)
```

**Torsion yoğunluğu:**
```
t(y) = w(y) × e(y) + m_pitch(y)
```

**Kesitteki torsion:**
```
T(y) = ∫_y^L_span t(ξ) dξ
```

### Bredt-Batho (Tek Hücre)

**Shear flow:**
```
q = T / (2 × A_m)
```

**Skin shear stress:**
```
τ_skin = q / t_skin
```

**Twist rate:**
```
dθ/dz = q × P / (2 × A_m × G × t)
```
burada P = hücre çevresi

### Spar (Dairesel Tüp)

**İç çap:**
```
d_i = d_o - 2t
```

**Kesit alanı:**
```
A = (π/4) × (d_o² - d_i²)
```

**Atalet momenti:**
```
I = (π/64) × (d_o⁴ - d_i⁴)
```

**Bending stress:**
```
σ_b = M × c / I
```
burada c = d_o/2 (dış fiber mesafesi)

**Shear stress:**
```
τ_spar = V / A
```

**Von Mises:**
```
σ_vm = √(σ_b² + 3 × τ²)
```

### Load Sharing

**Atalet bazlı yük paylaşımı:**
```
η_FS = I_FS / (I_FS + I_RS)
η_RS = I_RS / (I_FS + I_RS)
```

**Paylaşılan yükler:**
```
M_FS = η_FS × M
M_RS = η_RS × M
V_FS = η_FS × V
V_RS = η_RS × V
```

### Defleksiyon

**Moment-area yöntemi (sayısal):**
```
δ_tip = ∫_0^L_span (L_span - y) × M(y) / (E × I) dy
```

### Kabul Kriterleri

```
τ_skin ≤ τ_allow = τ_u / SF
σ_vm_spar ≤ σ_allow = σ_u / SF
|δ_tip| ≤ L_span / 20
X_FS% < X_RS%
t < d_o / 2
N_Rib ≥ 2
```

---

## ⚠️ Kabuller ve Sınırlamalar

### Temel Kabuller

1. **Tek kanat**: Hesaplar yarı-kanat (half-wing) içindir
2. **Statik analiz**: Dinamik/fatigue etkileri dahil değil
3. **C_L kullanılmıyor**: Lift = n × W_0 üzerinden hesaplanır
4. **Cantilever kiriş**: Root'ta ankastre, tip'te serbest
5. **Lineer taper**: Chord lineer olarak azalır
6. **Tek hücre wing-box**: Default olarak tek hücreli
7. **Sabit spar kesiti**: Span boyunca değişmiyor

### Sınırlamalar

| Sınırlama | Açıklama |
|-----------|----------|
| Buckling yok | Panel/spar burkulma analizi yok |
| FEM değil | Preliminary sizing aracı |
| Aeroelastik yok | Flutter/divergence analizi yok |
| Basit rib modeli | Rib stress yaklaşık |
| Airfoil yok | Gerçek airfoil konturu kullanılmıyor |

### Ne Zaman Kullanılmalı

✅ **Uygun:**
- Konsept tasarım aşaması
- Hızlı parametrik çalışmalar
- İlk boyutlandırma tahminleri
- Eğitim amaçlı

❌ **Uygun Değil:**
- Detaylı yapısal analiz
- Sertifikasyon hesapları
- Final tasarım doğrulaması

---

## 💻 Örnek Kullanım

### Temel Kullanım

```bash
python main.py
```

### Programatik Kullanım

```python
from materials import MaterialDatabase, MaterialSelection
from geometry import PlanformParams, SparPosition
from optimization import DesignSpace, DesignRange, run_optimization
from loads import FlightCondition, AeroCenter

# Planform tanımla
planform = PlanformParams.from_input(
    b=3.0,              # wingspan [m]
    AR=11.0,            # aspect ratio
    taper_ratio=0.45,   # taper ratio
    t_c=0.12,           # thickness ratio
    S_ref=3.17,         # ref area [m²]
    C_r_mm=600,         # root chord [mm]
    c_MGC=0.5,          # MGC [m]
    Y_bar_mm=500        # Y_bar [mm]
)

# Uçuş koşulu
flight = FlightCondition(
    W0=65,              # MTOW [N]
    n=2.0,              # load factor
    V_c=21,             # velocity [m/s]
    rho=1.773,          # air density [kg/m³]
    C_m=-0.003,         # pitching moment coef
    S_ref=3.17,
    c_MGC=0.5
)

# Malzemeler
db = MaterialDatabase()
materials = MaterialSelection.from_database(
    db,
    spar_key='CFRP_UD',
    skin_key='GFRP',
    rib_key='GFRP'
)

# Tasarım uzayı
design_space = DesignSpace(
    N_Rib=DesignRange(4, 8, 1),
    t_rib_mm=DesignRange(1.5, 2.5, 0.5),
    X_FS_percent=DesignRange(12, 18, 2),
    X_RS_percent=DesignRange(48, 72, 6),
    d_FS_outer_mm=DesignRange(16, 24, 2),
    t_FS_mm=DesignRange(0.8, 1.2, 0.2),
    d_RS_outer_mm=DesignRange(16, 24, 2),
    t_RS_mm=DesignRange(0.8, 1.2, 0.2),
)

# Optimizasyon
ac = AeroCenter(x_ac_percent=25, Lambda_ac_deg=0)
best, optimizer = run_optimization(
    planform, flight, ac, materials, design_space,
    t_skin_mm=0.625, SF=1.5
)

# Sonuçlar
print(f"Best mass: {best.mass_total * 1000:.2f} g")
```

---

## 📈 Çıktı Dosyaları

Program çalıştıktan sonra:

```
output_plots/
├── load_distributions_combined.png   # w(y), V(y), M(y)
├── torsion_combined.png              # T(y), q(y), τ(y)
├── spar_von_mises.png                # σ_vm(y) for FS & RS
├── twist_rate.png                    # dθ/dz(y)
└── planform.png                      # Wing top view

optimization_results.json             # All numerical results
```

---

## 🔧 Geliştirme

### Yeni Malzeme Ekleme

```python
from materials import MaterialDatabase, Material

db = MaterialDatabase()
db.add_material('TITANIUM', Material(
    name='Ti-6Al-4V',
    E=113.8e9,
    nu=0.342,
    density=4430,
    sigma_u=950e6,
    tau_u=550e6
))
```

### Custom Analiz

```python
from loads import analyze_loads, LoadDistributionType
from torsion import analyze_all_stations

# Elliptic yük ile analiz
loads = analyze_loads(
    y, chord, x_FS, x_RS, flight, ac, L_span,
    LoadDistributionType.ELLIPTIC
)

# Torsion analizi
torsion_results = analyze_all_stations(
    y, loads.T, A_m, t_skin, G, box_width, box_height
)
```

---

## 📚 Referanslar

1. Bruhn, E.F. - "Analysis and Design of Flight Vehicle Structures"
2. Niu, M.C.Y. - "Airframe Structural Design"
3. Megson, T.H.G. - "Aircraft Structures for Engineering Students"
4. Bredt-Batho Theory - Closed Section Torsion

---

## 📄 Lisans

MIT License

---

## 👥 Katkıda Bulunanlar

- Structural analysis formulations
- Grid search optimization
- Validation with Codex AI

---

**Not**: Bu araç preliminary sizing içindir. Kritik tasarım kararları için detaylı FEM analizi gereklidir.

## ✍️ İmza

Ayberk Cem Aksoy
Berke Tezgöçen
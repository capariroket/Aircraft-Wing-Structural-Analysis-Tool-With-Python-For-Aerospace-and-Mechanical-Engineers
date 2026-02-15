#!/usr/bin/env python3
"""
================================================================================
PROGRAM_DOCUMENTATION.py
================================================================================

YARI-KANAT (HALF-WING) YAPISAL BOYUTLANDIRMA PROGRAMI
Detayli Dokumantasyon ve Algoritma Aciklamasi

Bu dosya programin calisma mantigini, modullerini, formullerini ve
algoritmasini detayli olarak aciklar. Kod icermez, sadece dokumantasyondur.
Calistirdiginizda bu dokumantasyonu terminale basar.

Yazar: Otomatik olusturulmustur
================================================================================
"""


DESCRIPTION = """
Half-Wing Structural Sizing Program — Description (max 350 words)
=================================================================

This program performs preliminary structural sizing of a half-wing for
unmanned aerial vehicles (UAVs) or small aircraft. Given flight conditions
(MTOW, load factor, cruise speed), wing planform geometry (span, aspect ratio,
taper ratio, thickness-to-chord ratio), and material selections (separate
materials for spars, skin, and ribs), it finds the minimum-weight structural
configuration through a two-phase optimization process.

The wing structure consists of three main components: two circular-tube spars
(front and rear) carrying bending and shear loads, a thin skin shell providing
aerodynamic shape and resisting torsion-induced shear flows, and internal ribs
maintaining cross-sectional shape and preventing skin panel buckling.

Phase 1 (Fast Screening) evaluates thousands of design variable combinations
(spar diameters, wall thicknesses, spar positions, rib thickness) against
structural criteria: skin shear stress limits, spar bending-shear interaction,
spar cross-section adequacy, assembly clearance, and tip twist angle. Passing
configurations are ranked by total mass, and the lightest 20 advance to Phase 2.

Phase 2 (Buckling Fix) performs detailed panel buckling analysis using a
sweep-based adaptive rib placement algorithm. Starting from the wing root, it
uses binary search to find the maximum feasible rib spacing at each location
based on local shear and compression buckling criteria. This produces a
non-uniform rib distribution — closely spaced near the root where stresses are
highest, and widely spaced toward the tip. If needed, skin thickness is
incrementally increased. Finally, rib web shear buckling is checked at each
station, and only failing ribs are locally thickened in 0.5 mm increments.

Key analysis methods include Bredt-Batho closed-section torsion theory, beam
bending theory with transformed (modular ratio) box-beam sections, and
classical plate buckling formulas for skin panels and rib webs. The program
outputs detailed stress reports, bay-by-bay buckling tables, mass breakdowns,
comparison plots (Phase 1 vs Phase 2 planform), and exports results to JSON/CSV.

An optional genetic algorithm (pymoo) is also available as an alternative to
the grid search optimizer.
"""


def print_section(title: str, content: str):
    """Baslik ve icerik yazdir."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(content)


def main():
    print("""
################################################################################
##                                                                            ##
##   YARI-KANAT YAPISAL BOYUTLANDIRMA PROGRAMI                               ##
##   (Half-Wing Rib-Spar-Skin Preliminary Structural Sizing)                  ##
##                                                                            ##
##   Detayli Dokumantasyon                                                    ##
##                                                                            ##
################################################################################
""")

    # =========================================================================
    # 1. PROGRAM GENEL BAKIS
    # =========================================================================
    print_section("1. PROGRAM GENEL BAKIS", """
Bu program, bir yari-kanatin (half-wing) on-tasarim seviyesinde yapisal
boyutlandirilmasini yapar. Amac: verilen ucus kosullari, geometri ve malzeme
secenekleri icin en hafif yapisal konfigurasyonu bulmak.

Kanat Yapisi Bileşenleri:
  - SKIN (Kabuk): Kanatin dis yuzeyi. Aerodinamik profili saglar, 
    kesme ve basinca akislari tasir.
  - SPAR (Kiris): On spar (Front Spar, FS) ve arka spar (Rear Spar, RS).
    Dairesel boru kesitli. Egilme momentini ve kesme kuvvetini tasir.
  - RIB (Kaburga): Kanat kesitinin seklini koruyan ic yapilar.
    Skin panellerini destekler, burkulmayi onler.

Program Ciktilari:
  - Optimal spar cap/duvar kalinliklari
  - Optimal spar pozisyonlari (% kord)
  - Optimal rib sayisi ve pozisyonlari (adaptif)
  - Kutle dagilimi (skin, spar, rib)
  - Gerilme analizi (egilme, kesme, von Mises, burkulma)
  - Emniyet marjinlari
  - Grafik ve raporlar
""")

    # =========================================================================
    # 2. MODUL YAPISI
    # =========================================================================
    print_section("2. MODUL YAPISI VE VERI AKISI", """
Program 12 modulden olusur. Veri akisi su sekildedir:

  main.py (Ana Program - Kullanici Arayuzu)
    |
    v
  materials.py --> Malzeme ozellikleri (E, sigma_u, tau_u, rho, nu)
    |
    v
  geometry.py --> Kanat geometrisi (kord, spar pozisyonlari, kanat kutusu)
    |
    v
  loads.py --> Yuk dagilimi (w, V, M, T)
    |
    v
  torsion.py --> Burulma analizi (q, tau_skin, twist_rate)
    |
    v
  spars.py --> Spar gerilmeleri + Box Beam modeli (sigma_b, tau, sigma_vm)
    |
    v
  buckling.py --> Burkulma kritik gerilmeleri (tau_cr, sigma_cr, R_comb)
    |
    v
  ribs.py --> Adaptif rib yerlestirme + rib web burkulma kontrolu
    |
    v
  optimization.py / ga_optimization.py --> Optimizasyon (Grid Search / GA)
    |
    v
  reporting.py + plots.py --> Rapor ve grafikler

Her modul SI birim sistemi kullanir (m, Pa, kg, N).
Kullaniciya gosterimde mm, MPa, g kullanilir.
""")

    # =========================================================================
    # 3. MALZEME MODULU
    # =========================================================================
    print_section("3. MATERIALS.PY - Malzeme Veritabani", """
Amac: Malzeme ozelliklerini yonetir.

Mevcut Malzemeler:
  1. AL7075-T6  : E=71.7 GPa, sigma_u=572 MPa, tau_u=331 MPa, rho=2810 kg/m3
  2. AL2024-T3  : E=73.1 GPa, sigma_u=483 MPa, tau_u=283 MPa, rho=2780 kg/m3
  3. CFRP_UD    : E=135 GPa,  sigma_u=1500 MPa, tau_u=120 MPa, rho=1600 kg/m3
  4. GFRP       : E=40 GPa,   sigma_u=600 MPa,  tau_u=50 MPa,  rho=1900 kg/m3
  5. PLA        : E=3.5 GPa,  sigma_u=50 MPa,   tau_u=30 MPa,  rho=1250 kg/m3
  6. STEEL_4130 : E=205 GPa,  sigma_u=670 MPa,  tau_u=400 MPa, rho=7850 kg/m3

Her malzeme icin otomatik hesaplanan:
  G = E / (2 * (1 + nu))     [Pa]  (Kayma modulu)

Emniyet katsayisi (SF) ile izin verilen gerilmeler:
  sigma_allow = sigma_u / SF
  tau_allow   = tau_u / SF

Kullanici 3 farkli malzeme secer:
  - Spar malzemesi (genellikle CFRP veya Al)
  - Skin malzemesi (genellikle PLA veya GFRP)
  - Rib malzemesi  (genellikle PLA)
""")

    # =========================================================================
    # 4. GEOMETRI MODULU
    # =========================================================================
    print_section("4. GEOMETRY.PY - Kanat Geometrisi", """
Amac: Kanat planform geometrisini ve kanat kutusu kesit ozelliklerini hesaplar.

PLANFORM PARAMETRELERI (Giris):
  b      = Kanat acikligi [m]
  AR     = En-boy orani [-]
  lambda = Daralma orani [-]  (C_tip / C_root)
  t/c    = Kalinlik/kord orani [-]
  S_ref  = Referans kanat alani [m2]

HESAPLANAN PARAMETRELER:
  C_r (Kok kord):
    C_r = 2 * S_ref / (b * (1 + lambda))

  c_MGC (Ortalama geometrik kord):
    c_MGC = (4/3) * sqrt(S_ref/AR) * (1 + lambda + lambda^2) / (1 + 2*lambda + lambda^2)

  Y_bar (MGC spanwise pozisyonu):
    Y_bar = (b/6) * (1 + 2*lambda) / (1 + lambda)

  N_Rib (Minimum rib sayisi):
    N_Rib = ceil(1 + sqrt(AR * S_ref) / c_MGC)

  L_span = b / 2              (Yari-kanat acikligi)
  c_tip  = lambda * C_r       (Uc kord)

KORD DAGILIMI (Lineer daralma):
  c(y) = C_r - (C_r - c_tip) * (y / L_span)

  y=0'da c = C_r (kok), y=L_span'da c = c_tip (uc)

KANAT KUTUSU KESITI (her istasyonda):
  x_FS = X_FS% * c(y)         On spar pozisyonu [m]
  x_RS = X_RS% * c(y)         Arka spar pozisyonu [m]
  h_box = t/c * c(y)           Kutu yuksekligi [m]
  width = x_RS - x_FS          Kutu genisligi [m]
  A_m   = width * h_box         Kapali alan [m2] (Bredt-Batho icin)

SPAR GECIS ACISI (Sweep):
  Lamda_spar = arctan(tan(Lamda_ac) + 4*(x_ac - X_spar%)*(1-lambda) / (x_ac*AR*(1+lambda)))

SKIN ARC UZUNLUKLARI (Parabolik profil):
  Parabolik airfoil yaklasimi ile LE-FS, FS-RS bolgelerinin
  yuzey uzunluklari hesaplanir. Skin kutlesi icin kullanilir.

RIB ALANLARI:
  Dikdortgen profil: S_rib = h_box * x_FS  (veya x_RS - x_FS)
  Parabolik profil:  S_rib = (2/3) * x * h_box  (airfoil yaklasimi)
""")

    # =========================================================================
    # 5. YUK MODULU
    # =========================================================================
    print_section("5. LOADS.PY - Yuk Analizi", """
Amac: Kanat boyunca tasiyici yuk, kesme kuvveti, egilme momenti ve
burulma momentini hesaplar.

UCUS KOSULLARI:
  W0    = Azami kalkis agirligi [N]
  n     = Yuk faktoru [-] (manevra/gust)
  V_c   = Seyir hizi [m/s]
  rho   = Hava yogunlugu [kg/m3]
  C_m   = Hatve momenti katsayisi [-]
  S_ref = Referans alan [m2]
  c_MGC = Ortalama geometrik kord [m]

TOPLAM TASIYICI KUC:
  L_total = n * W0              [N]
  L_half  = L_total / 2         [N] (tek kanat)

TASIYICI KUC DAGILIMI w(y):
  1) Uniform:  w(y) = L_half / L_span
  2) Eliptik:  w(y) = w0 * sqrt(1 - (y/L_span)^2)
               w0 = 4 * L_half / (pi * L_span)

  Eliptik dagilim daha gercekci: kokte maksimum, ucta sifir.

KESME KUVVETI V(y):
  V(y) = integral_y^L_span w(xi) d_xi
  
  Uctan koke dogru entegrasyon. Ucta V=0, kokte V=L_half.

EGILME MOMENTI M(y):
  M(y) = integral_y^L_span V(xi) d_xi
  
  Ucta M=0, kokte maksimum.

HATVE MOMENTI DAGILIMI:
  M_pitch_total = C_m * q_inf * S_ref * c_MGC
  q_inf = 0.5 * rho * V_c^2

  Kord-agirlikli: m_pitch(y) = k * c(y)
  k belirlenir: integral m_pitch dy = M_pitch_half

BURULMA:
  Eksantrisite: e(y) = x_ac(y) - x_sc(y)
    x_ac = aerodinamik merkez (~%25 kord)
    x_sc = kesme merkezi = (x_FS + x_RS) / 2

  Burulma yogunlugu: t(y) = w(y) * e(y) + m_pitch(y)
  Burulma momenti:   T(y) = integral_y^L_span t(xi) d_xi
""")

    # =========================================================================
    # 6. BURULMA MODULU
    # =========================================================================
    print_section("6. TORSION.PY - Burulma ve Kayma Akisi", """
Amac: Bredt-Batho teorisi ile kapali kesitte burulma analizi.

BREDT-BATHO TEK HUCRE:
  Kayma akisi:    q = T / (2 * A_m)                [N/m]
  Kayma gerilmesi: tau_skin = q / t_skin            [Pa]
  Burulma orani:  d_theta/dz = q * P / (2 * A_m * G * t_skin)  [rad/m]

  Burada:
    T     = Burulma momenti [N.m]
    A_m   = Kapali alan [m2]
    t_skin = Kabuk kalinligi [m]
    P     = Cevre uzunlugu [m]
    G     = Kayma modulu [Pa]

UC BURULMA ACISI:
  theta_tip = integral_0^L_span (d_theta/dz) dy     [rad]
  
  Derece: theta_tip_deg = theta_tip * 180 / pi

  Kisitlama: theta_tip_deg <= theta_max (genellikle 2-10 derece)

KRITIK ISTASYON:
  tau_skin en yuksek olan istasyon (genellikle kok).
  Kokte T ve q en buyuk, skin en kalin (kord en genis).
""")

    # =========================================================================
    # 7. SPAR MODULU
    # =========================================================================
    print_section("7. SPARS.PY - Spar Analizi ve Box Beam Modeli", """
Amac: Spar gerilmelerini, yuk paylaşimini ve box beam modelini hesaplar.

SPAR GEOMETRISI (Dairesel Boru Kesit):
  d_outer = Dis cap [m]
  t_wall  = Duvar kalinligi [m]
  d_inner = d_outer - 2 * t_wall

  Alan:    A = (pi/4) * (d_outer^2 - d_inner^2)
  Atalet:  I = (pi/64) * (d_outer^4 - d_inner^4)
  c_dist:  c = d_outer / 2  (dis fiber mesafesi)
  J:       J = (pi/32) * (d_outer^4 - d_inner^4)

YUK PAYLASIMI (eta):
  On spar:  eta_FS = (X_RS% - x_ac%) / (X_RS% - X_FS%)
  Arka spar: eta_RS = (x_ac% - X_FS%) / (X_RS% - X_FS%)

  Ornek: X_FS=15%, X_RS=60%, x_ac=25%
    eta_FS = (60-25)/(60-15) = 0.778  (%77.8 yuk on spara)
    eta_RS = (25-15)/(60-15) = 0.222  (%22.2 yuk arka spara)

GERILME HESAPLARI:
  Egilme gerilmesi:  sigma_b = M * eta * c / I    [Pa]
  Kesme gerilmesi:   tau     = T * eta * r / J     [Pa]
  Von Mises:         sigma_vm = sqrt(sigma_b^2 + 3*tau^2)  [Pa]

BOX BEAM MODELI (Transformed Section / Donusturulmus Kesit):
  Spar ve skin farkli malzemeden olabilir. Modular oran:
    n = E_skin / E_spar

  Skin'in atalet momenti (Steiner teoremi):
    I_skin = n * [2 * w_box * t_skin * (h_box/2)^2 + 2 * (1/12) * t_skin * h_box^3]
    (Ust + alt skin flanslari)

  Toplam: I_box = I_FS + I_RS + I_skin

  Skin basinc gerilmesi:
    sigma_skin_comp = M * (h_box/2) * n / I_box

  Bu deger burkulma kontrolu icin kullanilir.

UC COKME (Tip Deflection):
  delta_tip = integral_0^L (L-y) * M(y) / (E * I_box) dy
  (Moment-alan metodu ile sayisal entegrasyon)

ETKLESIM FORMULU (Spar Kabulu/Reddi):
  (sigma_b / sigma_allow) + (tau / tau_allow)^2 < 1.0

  Bu formul bending ve shear'i birlikte degerlendirir.
  1.0'dan kucukse PASS, buyukse FAIL.

  ONEMLI: sigma_allow = sigma_u / SF ve tau_allow = tau_u / SF
  SF artinca her iki allowable duser, ozellikle tau terimi (kareli)
  cok hizli buyur.
""")

    # =========================================================================
    # 8. BURKULMA MODULU
    # =========================================================================
    print_section("8. BUCKLING.PY - Panel Burkulma Analizi", """
Amac: Skin panellerinin ve rib web'lerinin burkulma kontrolu.

SKIN KESME BURKULMASI:
  tau_cr = k_s * pi^2 * E_skin / (12 * (1 - nu^2)) * (t_skin / s)^2

  k_s = 10  (basit mesnetli uzun plaka, program varsayilani)
  s   = rib araligi [m]

  tau_panel > tau_cr ise panel BURKUL.

SKIN BASINC BURKULMASI:
  sigma_cr = k_c * pi^2 * E_skin / (12 * (1 - nu^2)) * (t_skin / s)^2

  k_c = 8  (basit mesnetli tek yonlu basinc, program varsayilani)

  sigma_panel > sigma_cr ise panel BURKUL.

BIRLESIK ETKILESIM (Combined Interaction):
  R = (tau / tau_cr)^2 + (sigma / sigma_cr)

  R < 1.0 --> PASS
  R >= 1.0 --> FAIL

  Bu formul kesme ve basinc burkulmalarinin birlikte etkisini
  degerlendirir. Tek basina gecen bir panel, birlesik yuklemede
  burkuabilir.

MAKSIMUM RIB ARALIGI:
  s_max_shear = t_skin * sqrt(k_s * pi^2 * E / (12*(1-nu^2) * tau_panel))
  s_max_comp  = t_skin * sqrt(k_c * pi^2 * E / (12*(1-nu^2) * sigma_comp))

RIB WEB BURKULMASI:
  tau_cr_rib = k_s_rib * pi^2 * E_rib / (12*(1-nu_rib^2)) * (t_rib / h_box)^2

  k_s_rib = 5.34 (basit mesnetli)
  tau_rib = V / (h_box * t_rib)  (rib web kayma gerilmesi)

  tau_rib > tau_cr_rib ise rib web'i BURKUL.

BURKULMA MODLARI:
  Mod 1: Sadece kesme burkulmasi (tau_cr)
  Mod 2: Kesme + Basinc burkulmasi (tau_cr + sigma_cr + R_comb)
""")

    # =========================================================================
    # 9. RIB MODULU
    # =========================================================================
    print_section("9. RIBS.PY - Rib Geometrisi ve Adaptif Yerlestirme", """
Amac: Rib geometrisi, kutle hesabi ve akilli rib yerlestirme algoritmasi.

RIB GEOMETRISI:
  Her rib bir istasyondaki kanat kesitini kaplar.
  
  Dikdortgen profil:
    S_LE_FS = h_box * x_FS                (On spar oncesi alan)
    S_FS_RS = h_box * (x_RS - x_FS)       (Sparlar arasi alan)
  
  Parabolik profil (airfoil yaklasimi):
    S_LE_FS = (2/3) * x_FS * h_box
    S_FS_RS = (2/3) * x_RS * h_box - S_LE_FS

RIB KUTLE HESABI:
  Tek rib: m_rib = S_total * t_rib * rho_rib
  Toplam:  m_ribs = sum(m_rib_i) for i = 1..N  (kok rib haric)

  Rib web burkulma fixi sonrasi her rib kendi t_rib'ine gore:
  m_ribs = sum(S_total_i * t_rib_i * rho_rib)

ADAPTIF RIB YERLESTIRME ALGORITMASI (Sweep-Based):
  ================================================
  Eski yontem: Esit aralikli riblerden baslayip boluyor (sorunluydu)
  Yeni yontem: Kokten uca supu tarzi yerlestirme

  Algoritma:
  1. y_left = 0 (kokten basla)
  2. Kalan mesafe [y_left, y_tip] tek bay olarak gecerse --> BITTI
  3. Binary search ile y_left'ten baslayarak en genis gecerli
     araligi bul (burkulma siniri)
  4. Yeni rib'i y_left + s_feasible'a koy
  5. y_left = y_left + s_feasible, adim 2'ye don

  Onemli ozellikler:
  - Yapay ust sinir (s_max) YOK -- sadece burkulma belirler
  - Kokte stresler yuksek --> aralik dar (~84mm)
  - Uca dogru stresler duser --> aralik genisler (~390mm)
  - %100 kaplama garanti

  Binary Search Detayi:
  Faz 1: Mesafeyi yarilayarak ilk gecerli araligi bul
  Faz 2: PASS ve FAIL arasinda %2 toleransla daralt

RIB WEB BURKULMA FIXI:
  Phase 2'den sonra her rib istasyonunda:
  1. tau_rib = V / (h_box * t_rib) hesapla
  2. tau_cr_rib = k_s * pi^2 * E / (12*(1-nu^2)) * (t_rib/h_box)^2
  3. Eger tau_rib > tau_cr_rib:
     t_rib'i 0.5mm artir, tekrar kontrol et
     Gecene kadar tekrarla (max 10mm'ye kadar)
  4. Sadece burkulan ribler kalinlasir, digerlerine dokunulmaz

  Ornek sonuc (t_rib_initial = 1mm):
    Kok bolge (0-537mm):   t_rib = 2.0mm  (V yuksek)
    Orta bolge (635-1490mm): t_rib = 1.5mm  (V orta)
    Uc bolge (1658-tip):    t_rib = 1.0mm  (V dusuk, degisiklik yok)
""")

    # =========================================================================
    # 10. OPTIMIZASYON
    # =========================================================================
    print_section("10. OPTIMIZATION.PY - Iki Fazli Optimizasyon", """
Amac: En hafif konfigurasyonu bulmak icin iki fazli arama.

TASARIM DEGISKENLERI (8 adet):
  1. N_Rib        : Rib bay sayisi (formulden hesaplanan)
  2. t_rib_mm     : Rib kalinligi [mm]
  3. X_FS_percent : On spar pozisyonu [% kord]
  4. X_RS_percent : Arka spar pozisyonu [% kord]
  5. d_FS_outer_mm: On spar dis capi [mm]
  6. t_FS_mm      : On spar duvar kalinligi [mm]
  7. d_RS_outer_mm: Arka spar dis capi [mm]
  8. t_RS_mm      : Arka spar duvar kalinligi [mm]

FAZ 1 - HIZLI KABUL (Burkulma Yok):
  =========================================
  Amac: Tum kombinasyonlari hizla tarayarak burkulma haric
  temel kriterleri saglayan konfigurasyonlari bulmak.

  Kontrol edilen kriterler (sirasyla):
  1. Geometri dogrulamasi (X_FS < X_RS, t < d/2, vb.)
  2. Montaj kontrolu (d_FS + 10mm < h_box, d_RS + 10mm < h_box)
  3. Skin kayma gerilmesi (tau_skin < tau_allow_skin)
  4. Uc burulma acisi (theta_tip < theta_max)
  5. On spar etkilesim: (sigma_b/sigma_allow) + (tau/tau_allow)^2 < 1.0
  6. Arka spar etkilesim: ayni formul
  7. On spar alan: A_Act_FS > A_Cri_FS
  8. Arka spar alan: A_Act_RS > A_Cri_RS

  Herhangi bir kriter basarisiz olursa --> RED (o kriterde durur)
  Tum kriterler gecerse --> KABUL, kutle hesaplanir

  Kutle hesabi:
    m_skin = S_skin * t_skin * rho_skin
    m_FS   = A_FS * L_FS * rho_spar (sweep acisi dahil)
    m_RS   = A_RS * L_RS * rho_spar
    m_ribs = sum(S_rib * t_rib * rho_rib)
    m_total = m_skin + m_FS + m_RS + m_ribs

  Sonuc: En hafif 20 konfigurasyon (Top-20) Faz 2'ye gider.

FAZ 2 - BURKULMA DUZELTME:
  =========================================
  Amac: Top-20 konfigurasyonun her biri icin burkulma kontrolu yapip
  rib araligini ve/veya skin kalinligini ayarlamak.

  Adimlar:
  1. N_Rib_min'den baslayarak adaptive_rib_insertion calistir
     (sweep-based algoritma ile optimal rib pozisyonlari bul)
  2. Eger gecerli (feasible) --> kutle hesapla
  3. Eger gecersiz VE buckling_mode=2 --> t_skin'i 0.3mm artir, tekrar dene
  4. En dusuk kutleyi veren (t_skin, rib_layout) kombinasyonunu sec
  5. Rib web burkulma kontrolu yap, gereken riblerde t_rib kalinlastir
  6. Kutleyi guncelle

  Burkulma modlari:
    Mod 1: Sadece kesme burkulmasi
    Mod 2: Kesme + basinc burkulmasi (daha katki, genellikle daha fazla rib)

  Sonuc: Her konfigurasyon icin final rib pozisyonlari, t_skin, t_rib,
  ve guncel kutle.

NIHAI DOGRULAMA (main.py'de):
  Faz 2 sonrasi en iyi cozum tekrar 8+ kriterle dogrulanir.
  Basarisiz olursa siradaki cozume gecilir.
""")

    # =========================================================================
    # 11. GA OPTIMIZASYON
    # =========================================================================
    print_section("11. GA_OPTIMIZATION.PY - Genetik Algoritma", """
Amac: Grid search'e alternatif olarak surekli tasarim uzayinda arama.

pymoo kutuphanesi ile Mixed-Variable GA kullanir.
Tek amac fonksiyonu: Toplam kutle minimizasyonu
10 esitsizlik kisitlamasi (g <= 0 = uygun):

  g[0]: tau_skin - tau_allow           (skin kayma)
  g[1]: FS_interaction - 1.0           (on spar etkilesim)
  g[2]: RS_interaction - 1.0           (arka spar etkilesim)
  g[3]: A_Cri_FS - A_Act_FS            (on spar alan)
  g[4]: A_Cri_RS - A_Act_RS            (arka spar alan)
  g[5]: d_FS + 10 - h_box              (on spar montaj)
  g[6]: d_RS + 10 - h_box              (arka spar montaj)
  g[7]: theta_tip - theta_max          (burulma)
  g[8]: sigma_comp - sigma_cr          (basinc burkulma)
  g[9]: R_combined - 1.0               (birlesik etkilesim)

Tipik ayarlar: pop_size=100, n_gen=150, ~15000 degerlendirme
""")

    # =========================================================================
    # 12. KUTLE HESAPLARI
    # =========================================================================
    print_section("12. KUTLE HESAPLARI DETAY", """
SKIN KUTLESI:
  S_skin = Kanat yari-yuzeyi alani (ust + alt skin, parabolik profil)
  m_skin = S_skin * t_skin * rho_skin

SPAR KUTLESI:
  A_spar = (pi/4) * (d_outer^2 - d_inner^2)
  L_spar = L_span / cos(Lambda_spar)    (sweep acisi etkisi)
  m_spar = A_spar * L_spar * rho_spar

  On ve arka spar ayri ayri:
  m_FS = A_FS * L_FS * rho_spar
  m_RS = A_RS * L_RS * rho_spar

RIB KUTLESI:
  Her rib istasyonunda:
    S_rib_i = RibGeometry.S_total (dikdortgen veya parabolik)
    t_rib_i = fix_rib_web_buckling sonrasi kalinlik [m]
    m_rib_i = S_rib_i * t_rib_i * rho_rib

  Toplam: m_ribs = sum(m_rib_i) for i = 1..N (kok haric)

  Not: Rib web burkulma fixi sonrasi kokteki riblar daha kalin
  olabilir (ornegin 2mm), uctakiler orijinal kalir (1mm).

TOPLAM KUTLE:
  m_total = m_skin + m_FS + m_RS + m_ribs
""")

    # =========================================================================
    # 13. PLANFORM PLOT
    # =========================================================================
    print_section("13. PLOTS.PY - Gorsellestime", """
Amac: Analiz sonuclarini grafik olarak gostermek.

URETILEN GRAFIKLER:
  1. load_distributions_combined.png
     - w(y): Tasiyici yuk dagilimi [N/m]
     - V(y): Kesme kuvveti [N]
     - M(y): Egilme momenti [N.m]

  2. torsion_combined.png
     - T(y): Burulma momenti [N.m]
     - q(y): Kayma akisi [N/m]
     - tau_skin(y): Skin kayma gerilmesi [MPa]

  3. spar_von_mises.png
     - sigma_vm_FS(y): On spar von Mises [MPa]
     - sigma_vm_RS(y): Arka spar von Mises [MPa]
     - sigma_allow: Izin verilen gerilme

  4. twist_rate.png
     - d_theta/dz(y): Burulma orani [deg/m]

  5. planform.png (CIFT GRAFIK):
     - Sol: Faz 1 (Uniform) - Esit aralikli rib dagilimi
     - Sag: Faz 2 (Adaptive) - Burkulma bazli optimal rib dagilimi
     - LE, TE, on spar, arka spar, riblar ve wing box gosterilir
""")

    # =========================================================================
    # 14. KABUL/RED KRITERLERI
    # =========================================================================
    print_section("14. KABUL / RED KRITERLERI OZET", """
Bir konfigurasyonun kabul edilmesi icin TUMU saglanmalidir:

  KRITER                              FORMUL                           LIMIT
  ------                              ------                           -----
  1. Skin kayma                       tau_skin < tau_allow_skin         tau_u/SF
  2. On spar etkilesim               (s_b/s_a) + (t/t_a)^2 < 1        1.0
  3. Arka spar etkilesim             (s_b/s_a) + (t/t_a)^2 < 1        1.0
  4. On spar alan                    A_Act_FS > A_Cri_FS               -
  5. Arka spar alan                  A_Act_RS > A_Cri_RS               -
  6. On spar montaj                  d_FS + 10 < h_box                 -
  7. Arka spar montaj                d_RS + 10 < h_box                 -
  8. Uc burulma                      theta_tip <= theta_max             2-10 deg
  9. Birlesik etkilesim (Faz 2)      R_combined < 1.0                  1.0
 10. Bay-by-bay burkulma (Faz 2)     Her bay gecmeli                   -
 11. Rib web burkulma (Faz 2)        tau_rib <= tau_cr_rib             -

ETKILESIM FORMULUNDEKI SF ETKISI:
  sigma_allow = sigma_u / SF
  tau_allow   = tau_u / SF

  Etkilesim = (sigma_b / (sigma_u/SF)) + (tau / (tau_u/SF))^2
            = (sigma_b * SF / sigma_u) + (tau * SF / tau_u)^2

  Sigma terimi SF ile LINEER artar.
  Tau terimi SF ile KARESEL artar!

  Bu yuzden yuksek SF degerlerinde tau terimi baskindirn ve
  daha buyuk spar kesitleri (d_outer, t_wall) gerekir.
""")

    # =========================================================================
    # 15. TIPIK IS AKISI
    # =========================================================================
    print_section("15. TIPIK IS AKISI (ORNEK)", """
1. Kullanici giris yapar:
   - Ucus kosullari (W0=650N, n=2, V_c=21m/s, ...)
   - Geometri (b=5.195m, AR=11, lambda=0.45, t/c=0.17)
   - Malzeme secimi (CFRP spar, PLA skin+rib)
   - Yapisal parametreler (t_skin=1.2mm, SF=1.0, ...)
   - Tasarim araliklari (d_FS: 16-30mm, t_FS: 0.5-3mm, ...)

2. Program otomatik hesaplar:
   - C_r = 842mm, c_MGC = 685mm, L_span = 2598mm, N_Rib = 12

3. FAZ 1 calisir:
   - Tum kombinasyonlar degerlendirilir (tipik: 5000-50000)
   - Her biri icin: geometri -> yukler -> burulma -> spar gerilmeleri -> kutle
   - Gecen konfigurasyonlar kutleye gore siralanir
   - En hafif 20 tanesi secilir

4. FAZ 2 calisir:
   - Her Top-20 konfigurasyon icin:
     a) Sweep-based rib yerlestirme (burkulma sinirinda)
     b) Gerekirse t_skin artirma (0.3mm adimlarla)
     c) Rib web burkulma kontrolu ve t_rib ayarlama
     d) Guncel kutle hesabi

5. En iyi cozum secilir, nihai dogrulama yapilir.

6. Sonuclar:
   - Konsol raporu (tum gerilmeler, marjinlar, kutleler)
   - Bay-by-bay burkulma tablosu
   - Rib web burkulma tablosu
   - Planform karsilastirma grafigi (Faz1 vs Faz2)
   - JSON ve CSV dosyalari
""")

    # =========================================================================
    # 16. DOSYA CIKTILARI
    # =========================================================================
    print_section("16. DOSYA CIKTILARI", """
Program asagidaki dosyalari uretir:

  optimization_results.json
    - Tum sonuclar JSON formatinda (optimal konfig, kutleler,
      gerilmeler, burkulma, optimizasyon gecmisi)

  bay_results.csv
    - Bay-by-bay burkulma analizi sonuclari (her bay icin
      aralik, gerilmeler, kritik degerler, marjinlar, PASS/FAIL)

  output_plots/
    - load_distributions_combined.png
    - torsion_combined.png
    - spar_von_mises.png
    - twist_rate.png
    - planform.png (Faz 1 vs Faz 2 karsilastirma)
""")

    print("\n" + "#"*80)
    print("##  DOKUMANTASYON SONU")
    print("#"*80)


if __name__ == "__main__":
    main()

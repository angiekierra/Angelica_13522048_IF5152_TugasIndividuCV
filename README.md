# Tugas Individu IF5152 – Visi Komputer

**Nama:** Angelica Kierra Ninta Gunting
**NIM:** 13522048

Repositori ini berisi implementasi empat fitur utama pada pengolahan citra digital menggunakan Python dan OpenCV, serta laporan hasil analisisnya.

---

## Struktur Folder

```
Nama_NIM_IF5152_TugasIndividuCV/
├── 01_filtering/         # Kode filtering (Gaussian & Median), output gambar hasil, tabel parameter
├── 02_edge/              # Kode deteksi tepi (Sobel & Canny), output edge map, tabel threshold
├── 03_featurepoints/     # Kode deteksi fitur (Harris, SIFT, FAST), output marking dan statistik hasil
├── 04_geometry/          # Kode transformasi geometri (Affine & Perspective), output overlay & matriks
├── 05_laporan.pdf        # Laporan lengkap hasil eksperimen
└── README.md             # Petunjuk instalasi dan cara menjalankan program
```

---

## Instalasi dan Persiapan

1. Buat virtual environment (opsional namun disarankan):

   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate        # Windows
   ```

2. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Struktur folder personal:

   * Untuk pengujian citra pribadi, buat folder bernama `personal/` di dalam masing-masing direktori fitur.
   * Letakkan file `.jpg`, `.jpeg`, atau `.png` di dalam folder tersebut.

---

## Cara Menjalankan Setiap Fitur

**Catatan:** Pastikan Anda masuk ke folder fitur terlebih dahulu sebelum menjalankan program.

### 1. Image Filtering

**Folder:** `01_filtering/`
**File:** `filtering.py`
**Deskripsi:** Mengimplementasikan Gaussian dan Median Filtering dengan variasi parameter kernel dan sigma.

**Cara menjalankan:**

```bash
cd 01_filtering
python filtering.py
```

**Langkah di terminal:**

* Pilih dataset: preset (cameraman, coins, dll) atau personal.
* Pilih jenis eksperimen:

  1. Gaussian (ubah sigma, kernel tetap)
  2. Gaussian (ubah kernel, sigma tetap)
  3. Median (ubah kernel)
* Hasil akan tersimpan di folder `outputs/` beserta file CSV parameter.

---

### 2. Edge Detection

**Folder:** `02_edge/`
**File:** `edge_detection.py`
**Deskripsi:** Mendeteksi tepi menggunakan metode Sobel dan Canny dengan variasi kernel serta threshold.

**Cara menjalankan:**

```bash
cd 02_edge
python edge_detection.py
```

**Langkah di terminal:**

* Pilih dataset: preset atau personal.
* Pilih metode:

  1. Sobel
  2. Canny
  3. Keduanya
* Masukkan nilai kernel atau threshold sesuai prompt.
* Output disimpan di `outputs/params_edge_*.csv` dan gambar hasil di folder `outputs/`.

---

### 3. Feature Points

**Folder:** `03_featurepoints/`
**File:** `featurepoints.py`
**Deskripsi:** Deteksi titik fitur menggunakan metode Harris, SIFT, dan FAST dengan pencatatan statistik hasil (jumlah titik, rata-rata response, ukuran, dan total response).

**Cara menjalankan:**

```bash
cd 03_featurepoints
python featurepoints.py
```

**Output:**

* Gambar marking titik fitur (`outputs/output_*.png`)
* Statistik hasil dalam bentuk CSV (`outputs/stats_featurepoints.csv`)

---

### 4. Geometric Transformation

**Folder:** `04_geometry/`
**File:** `geometry.py`
**Deskripsi:** Menerapkan transformasi Affine dan Perspective pada citra serta menampilkan efek perbedaan transformasi.

**Cara menjalankan:**

```bash
cd 04_geometry
python geometry.py
```

**Output:**

* Gambar hasil transformasi (overlay atau warp)
* File CSV berisi parameter matriks transformasi

---

## Laporan

File `05_laporan.pdf` berisi:

* Penjelasan teori dasar setiap fitur
* Analisis parameter dan hasil eksperimen
* Komparasi antar metode
* Refleksi pribadi

---

## Ringkasan Fitur Unik

* Output otomatis dalam folder `outputs/` dengan label teks yang jelas
* Penyimpanan parameter eksperimen ke file `.csv` untuk dokumentasi hasil
* Mendukung dataset preset dari `scikit-image` maupun citra pribadi
* Eksperimen interaktif melalui input terminal
* Struktur modular: setiap fitur dapat dijalankan secara independen


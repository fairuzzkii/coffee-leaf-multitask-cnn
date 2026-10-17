# Coffee Leaf Multi-Task CNN (Lightweight, Mobile/AR-Ready)

Implementasi model **Multi-Task Convolutional Neural Network (CNN) Lightweight** untuk diagnosis penyakit daun kopi, estimasi tingkat keparahan (*severity*), dan lokalisasi *bounding box* daun berbasis mobile (siap diintegrasikan ke **Unity Sentis** untuk Augmented Reality).

Proyek ini merupakan bagian dari penelitian skripsi dan Hibah Dosen Pemula (Fasilkom Universitas Jember, TKT 4):
> *"Model Visualisasi Interaktif Augmented Reality Berbasis Deep Learning untuk Diagnosis Dini dan Klasifikasi Tingkat Keparahan Penyakit Tanaman Kopi"*

---

## Ringkasan Arsitektur Model

Satu model terpadu (*single forward-pass*) dengan **Shared Backbone** menghasilkan **3 Output sekaligus**:

```
Citra Daun Kopi (224x224 RGB)
              │
Shared Backbone CNN (MobileNetV2 / EfficientNet-B0)
              │────────────────────────┬────────────────────────┐
      Head 1: Penyakit         Head 2: Keparahan        Head 3: Lokalisasi AR
       (FC + Softmax)           (FC + Sigmoid)              (FC + Sigmoid)
              │                        │                        │
       [1, 5] Probabilitas       [1, 1] Rasio Kontinu     [1, 4] Bounding Box
      (5 Kelas Penyakit)             (0.0 - 1.0)           [cx, cy, w, h] (0 - 1)
```

1. **Task 1: Diagnosis Penyakit** (5 Kelas: `Healthy`, `Miner`, `Leaf rust`, `Phoma`, `Cercospora`)
2. **Task 2: Tingkat Keparahan** (Nilai kontinu rentang $0.0 - 1.0$ / $0\% - 100\%$)
3. **Task 3: Lokalisasi Daun untuk AR** (Koordinat $[cx, cy, w, h]$ ternormalisasi $0 - 1$)

---

## Struktur Folder Dataset

Karena ukuran dataset citra mencapai puluhan ribu foto, dataset disimpan secara terpisah dan **tidak di-commit ke Git**.

Letakkan folder dataset **bersebelahan / dalam satu folder induk** dengan folder repository `coffee-leaf-multitask-cnn`:

```text
Penelitian_Kopi/                       ← (Folder Induk)
├── JMUBEN/                            ← Folder Dataset JMUBEN (Leaf rust, Phoma, Cercospora)
├── JMUBEN V2/                         ← Folder Dataset JMUBEN V2 (Healthy, Miner)
├── BRACOL_coffee_leaf_ images_datasets/ ← Folder Dataset BRACOL (daun + label keparahan)
└── coffee-leaf-multitask-cnn/         ← Repository ini
    └── classification/
        ├── main.py
        ├── export_onnx.py
        └── ...
```

---

## 💻 Kebutuhan Sistem & Instalasi (Requirements)

Sebelum menjalankan skrip, pastikan dependensi berikut telah terpasang di environment Python Anda:

```bash
pip install torch torchvision
pip install pillow pandas scikit-learn tqdm numpy
pip install onnx onnxscript onnxruntime
```
> **Catatan:** `onnx`, `onnxscript`, dan `onnxruntime` wajib dipasang untuk mengekspor dan menjalankan simulasi inferensi model untuk Unity Sentis.

---

## 🚀 Panduan Menjalankan di Terminal (VS Code / Lokal)

### 1. Masuk ke Direktori Proyek
Buka terminal di VS Code, lalu arahkan ke folder `classification`:
```bash
cd coffee-leaf-multitask-cnn/classification
```

### 2. Persiapan Dataset Gabungan (CSV)
Jalankan skrip berikut sekali untuk menggabungkan indeks ketiga dataset menjadi file CSV:
```bash
python prepare_combined_dataset.py
```
*(File CSV akan otomatis dibuat di folder `dataset/dataset_combined_balanced.csv`)*

---

### 3. Training Model

#### A. Training Multi-Task (MobileNetV2)
```bash
python main.py --train \
  --backbone mobilenet \
  --mode multi \
  --dataset combined \
  --csv_file dataset/dataset_combined_balanced.csv \
  --epochs 20 \
  --batch_size 24 \
  --lr 0.01 \
  --experiment_name exp_multitask_mobilenet
```

#### B. Training Multi-Task (EfficientNet-B0)
```bash
python main.py --train \
  --backbone efficientnet \
  --mode multi \
  --dataset combined \
  --csv_file dataset/dataset_combined_balanced.csv \
  --epochs 20 \
  --batch_size 24 \
  --lr 0.01 \
  --experiment_name exp_multitask_efficientnet
```

#### C. Training Single-Task (Baseline Pembanding Skripsi)
Jika ingin melatih model terpisah per task untuk data perbandingan di Bab 4:
```bash
# Contoh: Single-task Diagnosis Penyakit saja
python main.py --train \
  --backbone mobilenet \
  --mode single \
  --task disease \
  --dataset combined \
  --csv_file dataset/dataset_combined_balanced.csv \
  --epochs 20 \
  --experiment_name exp_singletask_disease_mobilenet
```

---

### 4. Ekspor ke Format ONNX (Siap Pakai di Unity Sentis)

Konversi bobot model PyTorch (`.pth`) ke format ONNX dengan output bernama:
```bash
# Ekspor MobileNetV2
python export_onnx.py \
  --weights results/exp_multitask_mobilenet/best_weights.pth \
  --output results/model_unity_mobilenet.onnx \
  --backbone mobilenet \
  --opset 12

# Ekspor EfficientNet-B0
python export_onnx.py \
  --weights results/exp_multitask_efficientnet/best_weights.pth \
  --output results/model_unity_efficientnet.onnx \
  --backbone efficientnet \
  --opset 12
```
Perintah di atas otomatis membuat file metadata: `model_unity_mobilenet_meta.json`.

---

### 5. Verifikasi Numerik Hasil Ekspor
Pastikan hasil prediksi ONNX sama persis dengan PyTorch:
```bash
python verify_export.py \
  --onnx results/model_unity_mobilenet.onnx \
  --weights results/exp_multitask_mobilenet/best_weights.pth \
  --backbone mobilenet
```

---

### 6. Simulasi Deteksi Citra & Output untuk Unity AR
Uji model pada sebuah foto daun kopi untuk melihat hasil diagnosis, keparahan, dan konversi koordinat AR:
```bash
python simulate_unity_inference.py \
  --image "../../JMUBEN/Phoma/41.jpg" \
  --onnx results/model_unity_mobilenet.onnx \
  --meta results/model_unity_mobilenet_meta.json
```

Contoh output JSON yang dihasilkan untuk tim Unity:
```json
{
  "source_image": "41.jpg",
  "diagnosis": {
    "predicted_class_id": 3,
    "disease_name": "Phoma",
    "confidence": 0.8845
  },
  "severity": {
    "raw_ratio": 0.352,
    "percentage": "35.2%"
  },
  "localization": {
    "unity_viewport_coordinates_bottom_left_origin": {
      "center_x": 0.4982,
      "center_y": 0.488,
      "width": 0.784,
      "height": 0.811
    }
  }
}
```

---

## Menjalankan di Google Colab (GPU T4)

Jika ingin melatih di Google Colab:
1. Upload folder dataset (dalam format `.zip`) ke Google Drive Anda.
2. Buka notebook [`classification/training_colab.ipynb`](classification/training_colab.ipynb).
3. Pastikan memilih runtime: **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
4. Ikuti langkah berurutan di dalam notebook (Mount Drive, Unzip, Training, Export ONNX, & Download).

---

## Kontrak Output untuk Unity Sentis

| Nama Output | Shape | Tipe Data | Keterangan |
|---|---|---|---|
| `disease_probs` | `[1, 5]` | `float32` | Probabilitas tiap kelas (Softmax). Argmax = kelas prediksi. |
| `severity` | `[1, 1]` | `float32` | Rasio keparahan kontinu ($0.0 - 1.0$). Kalikan 100 untuk $\%$. |
| `bbox` | `[1, 4]` | `float32` | Koordinat $[cx, cy, w, h]$ ternormalisasi ($0.0 - 1.0$). Origin model kiri-atas $\to$ konversi Unity: $y_{\text{unity}} = 1.0 - y_{\text{model}}$. |
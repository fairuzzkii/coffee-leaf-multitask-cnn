<USER_REQUEST>
# Project Context: Coffee Leaf Multi-Task CNN (Lightweight, Mobile/AR-Ready)



> File ini adalah konteks utama proyek untuk AI agent. Baca seluruh file ini sebelum mengubah kode apa pun. Kalau ada instruksi yang bertentangan dengan file ini, tanyakan dulu ke saya.



## 1. Ringkasan Proyek



Ini adalah **bagian Model AI** dari proyek besar:



> **Model Visualisasi Interaktif Augmented Reality Berbasis Deep Learning untuk Diagnosis Dini dan Klasifikasi Tingkat Keparahan Penyakit Tanaman Kopi** (Hibah Dosen Pemula, Fasilkom Universitas Jember, TKT 4).



Sistem akhirnya adalah aplikasi mobile (kamera) yang mendeteksi penyakit daun kopi, tingkat keparahannya, **dan posisi daun di frame kamera**, lalu menampilkannya lewat overlay AR real-time di **Unity**. Karena overlay AR butuh tahu *di mana* menaruh label, model harus mengeluarkan **koordinat (x, y)** selain klasifikasi dan severity. Repo ini **hanya mengerjakan model deep learning-nya** (training, evaluasi, ekspor), bukan aplikasi AR-nya, tapi **output model harus siap pakai di Unity** (lihat bagian 3.6).



Judul skripsi saya: *Implementasi Model Multi-Task CNN Lightweight untuk Diagnosis Penyakit Daun Kopi Berbasis Mobile.*



## 2. Asal Repo



- Repo ini di-**clone/fork dari** `https://github.com/esgario/lara2018.git` (Esgario et al., 2020, *Deep learning for classification and severity estimation of coffee leaf biotic stress*, Computers and Electronics in Agriculture, vol. 169).

- Repo asli itu adalah **baseline/referensi**: sudah punya konsep multi-task (klasifikasi penyakit + estimasi keparahan) pada dataset BRACOL.

- Repo saya: `https://github.com/fairuzzkii/coffee-leaf-multitask-cnn.git`

- Tujuan saya bukan menyalin mentah, tapi **memodernisasi dan memperluas**: backbone lebih ringan, dataset lebih besar/beragam, siap diekspor ke mobile.

- Saat mengubah kode warisan lara2018, jaga agar perubahan tetap bisa dijelaskan (saya perlu mempertanggungjawabkannya di skripsi). Jangan hapus file/fungsi asli tanpa alasan jelas; lebih baik tambah modul baru atau refactor terukur.



## 3. Konsep Model (WAJIB DIPATUHI)



### 3.1 Multi-task (3 output)

Satu model, satu forward pass, **tiga output sekaligus**:



| Task | Output | Tipe |

|------|--------|------|

| Task 1: Diagnosis penyakit | Kelas penyakit daun kopi | Klasifikasi multi-kelas |

| Task 2: Estimasi tingkat keparahan | Nilai tingkat keparahan (persentase) | **Regresi** |

| Task 3: Lokalisasi (untuk AR) | Koordinat daun/area terinfeksi: `x, y, w, h` (ternormalisasi 0-1) | **Regresi** |



Arsitektur (diagram skripsi/proposal + head lokalisasi untuk AR):



```

Citra daun kopi

      |

Shared Backbone CNN Lightweight  (MobileNet / EfficientNet)

      |------------------------|---------------------------|

Head Klasifikasi          Head Keparahan             Head Lokalisasi

(FC + Softmax)            (FC, 1 output,             (FC, 4 output: x, y, w, h,

                           TANPA softmax)             sigmoid -> rentang 0-1)

      |                        |                           |

Jenis penyakit           Tingkat keparahan          Posisi di frame (untuk overlay AR)

```



- **Satu shared backbone** dipakai ketiga task. Backbone TIDAK diduplikasi per task.

- **Head penyakit**: Fully Connected + Softmax, loss klasifikasi (mis. Cross-Entropy).

- **Head keparahan**: Fully Connected dengan satu output kontinu, **tanpa softmax**, loss regresi (mis. MSE/Huber/L1). Kalau target dinormalisasi ke rentang 0-1, boleh pakai sigmoid di output, tapi catat dan dokumentasikan keputusannya.

- **Head lokalisasi**: Fully Connected dengan 4 output `(x, y, w, h)`, sigmoid agar nilainya 0-1, loss regresi (mis. Smooth L1, opsional ditambah GIoU/IoU loss). `(x, y)` = **titik tengah** kotak, `(w, h)` = lebar dan tinggi kotak, semuanya relatif terhadap ukuran citra input.

- **Loss gabungan**: `Total Loss = λ1 * L_penyakit + λ2 * L_keparahan + λ3 * L_lokasi`. Nilai λ1, λ2, λ3 dibuat konfigurable, bukan hardcode.

- Label keparahan harus numerik. Kalau suatu dataset hanya punya kelas keparahan diskrit (bukan persentase), **jangan mengarang konversi**; tanyakan dulu ke saya cara memetakannya.

- **Batasan yang disengaja (scope TKT 4):** model ini memprediksi **satu daun/satu kotak per frame**, bukan banyak objek sekaligus. Jangan mengganti arsitektur menjadi detector penuh (YOLO/SSD/Faster R-CNN) tanpa persetujuan saya. Kalau menurut agent pendekatan ini terlalu lemah untuk AR, sampaikan sebagai saran, jangan langsung diubah.



### 3.2 Pembanding: Single-task vs Multi-task

Untuk membuktikan manfaat multi-task, saya perlu **baseline single-task** sebagai pembanding:



- **Single-task**: model terpisah per task. Model 1 = backbone + head klasifikasi penyakit. Model 2 = backbone + head estimasi keparahan. Model 3 = backbone + head lokalisasi. Masing-masing punya backbone sendiri.

- **Multi-task**: satu model terpadu (shared backbone + tiga head), seperti di 3.1.



Aturan perbandingan:

- Backbone, dataset, split, augmentasi, dan hyperparameter dasar **harus identik** antara single-task dan multi-task agar adil.

- Matriks eksperimen: **2 backbone (MobileNet, EfficientNet) x 2 skema (single-task, multi-task)**.

- Bandingkan akurasi/error, jumlah parameter total, ukuran model, dan latensi inferensi. Hipotesis yang diuji: multi-task lebih ringan dan lebih cepat (satu backbone, satu forward pass, bukan tiga model) dengan performa sebanding atau lebih baik.

- Skema dipilih lewat config/argumen (mis. `--mode single|multi`).



### 3.3 Dua kandidat backbone

Hanya dua yang dibandingkan secara resmi:



1. **MobileNet** (utamakan MobileNetV2/V3 yang tersedia di framework)

2. **EfficientNet** (utamakan varian kecil seperti EfficientNet-B0)



Jangan menambah backbone lain (ResNet, VGG, dll.) kecuali saya minta eksplisit. Kedua kandidat harus bisa dipilih lewat **satu config/argumen** (mis. `--backbone mobilenet|efficientnet`) dengan pipeline training/evaluasi yang identik, supaya perbandingannya adil.



### 3.4 Lightweight untuk mobile

Model akhir akan dijalankan di smartphone (edge device), jadi:



- Ukuran model dan latensi inferensi **sama pentingnya dengan akurasi**.

- Hindari operasi/layer yang sulit dikonversi ke format mobile.

- Catat untuk setiap eksperimen: jumlah parameter, ukuran file model, dan kecepatan inferensi, selain metrik akurasi.

- Input resolusi tetap dan sederhana (mis. 224x224), preprocessing harus **sama persis** antara training, inferensi di Python, dan inferensi di Unity.

- Resize dilakukan **langsung ke ukuran input tanpa crop** (boleh stretch), supaya koordinat ternormalisasi tetap bisa dipetakan balik ke frame kamera penuh. Kalau agent memilih letterbox/padding, itu harus didokumentasikan di kontrak output (3.6).



### 3.5 Target ekspor

- **ONNX adalah format utama untuk Unity.** AR dikerjakan di Unity dengan **Unity Sentis** (pengganti Barracuda), yang mengimpor ONNX. Pastikan graph hanya memakai operator yang didukung Sentis; cek dokumentasi Sentis untuk versi yang dipakai (termasuk opset ONNX yang didukung), jangan menebak.

- **TFLite** tetap diekspor sebagai format sekunder (mis. untuk aplikasi mobile Flutter).

- Pipeline setelah model: kamera → preprocess → inferensi → parse output → overlay AR.

- Karena itu, **format output model harus jelas dan stabil** dan didokumentasikan sebagai kontrak (3.6).



### 3.6 Kontrak Output untuk Unity (WAJIB)

Model ONNX harus punya **output bernama** (bukan satu tensor campuran), sehingga mudah di-parse di C#:



| Nama output | Shape | Isi |

|-------------|-------|-----|

| `disease_probs` | `[1, N_kelas]` | Probabilitas tiap kelas (sudah softmax). `argmax` = kelas prediksi |

| `severity` | `[1, 1]` | Tingkat keparahan. Dokumentasikan satuannya (0-1 atau 0-100) dan lakukan konsisten |

| `bbox` | `[1, 4]` | `(x, y, w, h)` ternormalisasi 0-1, `(x, y)` = titik tengah |



Aturan kontrak:

- **Konvensi koordinat model:** origin di **pojok kiri-atas** citra, sumbu x ke kanan, sumbu y ke bawah. Catatan untuk sisi Unity: koordinat layar/viewport Unity berorigin kiri-bawah, jadi `y_unity = 1 - y_model`. Tulis catatan ini di dokumentasi ekspor.

- **Input:** nama, shape (mis. `[1, 224, 224, 3]` atau `[1, 3, 224, 224]`, sesuai framework), tipe `float32`, **urutan channel (RGB)** dan **normalisasi** (mean/std atau 0-1) harus tertulis eksplisit.

- Simpan **`model_meta.json`** di samping file model, isinya: nama input/output, shape, urutan kelas (label map: indeks → nama penyakit), normalisasi, satuan severity, konvensi koordinat, nama backbone, dan versi/tanggal ekspor. File ini yang akan dipakai tim Unity.

- **Verifikasi ekspor wajib:** setelah ekspor, jalankan model ONNX dengan `onnxruntime` pada beberapa citra uji dan pastikan ketiga output **sama dengan model asli** dalam toleransi numerik kecil. Sediakan skrip kecil yang meniru preprocessing + parsing output seperti yang akan dilakukan Unity, plus contoh output (JSON) untuk beberapa citra sebagai referensi tim Unity.

- Jangan mengubah nama output, urutan kelas, atau konvensi koordinat setelah kontrak dibekukan tanpa memberi tahu saya, karena itu merusak integrasi Unity.



## 4. Dataset



| Dataset | Peran |

|---------|-------|

| **BRACOL** | 1.747 citra daun kopi arabika, punya label penyakit + keparahan (dataset asli Esgario et al.) |

| **JMuBEN** | 58.405 citra arabika, 5 kelas penyakit |

| **JMuBEN v2** | Versi lanjutan JMuBEN |



Catatan penting:

- Ketiga dataset punya **skema label berbeda**. Tantangan utama adalah **penyelarasan anotasi** agar semuanya kompatibel untuk pembelajaran multi-task (kelas penyakit + keparahan).

- Jangan berasumsi dataset punya label keparahan. Cek dulu; kalau suatu dataset tidak punya label keparahan, **tanyakan ke saya** cara menanganinya (mis. masking loss task keparahan, atau dataset itu hanya dipakai untuk task penyakit). Jangan mengarang label.

- Pastikan **tidak ada data leakage**: split train/val/test dilakukan di level citra/daun asli, sebelum augmentasi, dan augmentasi hanya di train set.

- **Label koordinat (bbox) belum tentu ada.** Setahu saya BRACOL dan JMuBEN tidak menyediakan anotasi bounding box. **Cek dulu** isi dataset (dan dokumentasinya) sebelum berasumsi. Kalau tidak ada, jangan mengarang: usulkan cara membuat bbox, misalnya otomatis dari segmentasi daun terhadap latar (thresholding/GrabCut/SAM) yang dicek visual pada sampel, lalu **minta persetujuan saya** sebelum memakainya sebagai label. Simpan bbox hasil generate sebagai file anotasi terpisah (jangan menimpa data asli) dan catat metodenya, karena harus bisa dijelaskan di skripsi.

- Ketiga task bisa punya cakupan label berbeda per dataset. Gunakan **masking loss per task** (loss task yang labelnya tidak ada tidak dihitung untuk sampel itu).

- **Augmentasi geometris** (flip, rotate, crop, scale) harus ikut mengubah bbox. Pastikan transformasi citra dan bbox sinkron, dan cek visual dengan menggambar bbox pada beberapa sampel augmentasi.

- Dataset besar **tidak di-commit ke git**. Gunakan `.gitignore` dan path yang bisa dikonfigurasi.



## 5. Metrik dan Target



Dari proposal (indikator ketercapaian):



- Klasifikasi penyakit: akurasi **> 90%**

- Estimasi keparahan (regresi): metrik utama **MAE dan RMSE** (juga R2 bila relevan). Catatan: proposal menulis "akurasi > 90%" untuk keparahan. Karena head keparahan adalah regresi, sebagai pelengkap **opsional** laporkan juga akurasi setelah nilai keparahan di-bin ke kelas (bin harus mengikuti definisi kelas dataset, jangan mengarang batas). Tanyakan saya kalau batas kelas belum jelas.

- Lokalisasi (bbox): metrik **mean IoU** dan **center error** (jarak titik tengah prediksi vs label, ternormalisasi). Belum ada target angka di proposal; usulkan target realistis setelah baseline jadi, jangan mengarang.

- Model berhasil dikonversi ke ONNX (utama) dan TFLite, dan jalan cepat di edge device

- Metrik penyakit: **Accuracy, Precision, Recall, F1-Score** (per kelas + macro) dan confusion matrix. Metrik keparahan: MAE, RMSE. Metrik lokalisasi: mean IoU, center error. Plus ukuran model dan kecepatan inferensi

- Laporkan hasil dalam tabel yang bisa langsung masuk ke skripsi: 2 backbone x 2 skema (single-task vs multi-task)



## 6. Di Luar Lingkup Repo Ini



Jangan membuat atau mengubah hal-hal ini kecuali saya minta:



- Aplikasi AR di Unity (dikerjakan terpisah)

- Aplikasi mobile Flutter / backend

- UCD, wawancara, SUS, expert review, black-box testing

- Backbone selain MobileNet dan EfficientNet



Tapi **desain model harus tetap ramah ke sana**: ekspor ONNX bersih, output bernama sesuai kontrak 3.6, `model_meta.json` lengkap, dan ringan.



## 7. Aturan Kerja untuk Agent



1. **Pahami dulu, baru ubah.** Baca struktur repo dan kode lara2018 yang ada sebelum membuat rencana.

2. **Rencana dulu untuk perubahan besar.** Untuk refactor/perubahan lintas banyak file, tulis rencana singkat dan minta konfirmasi sebelum mengeksekusi.

3. **Reproducibility.** Set seed, simpan config tiap eksperimen, simpan checkpoint dan log metrik terstruktur (mis. CSV/JSON) agar hasil bisa dimasukkan ke skripsi.

4. **Config-driven.** Backbone, dataset, hyperparameter, bobot loss diatur lewat config/argumen, bukan hardcode.

5. **Jangan mengarang hasil.** Jangan menulis angka akurasi/latensi yang tidak benar-benar dijalankan. Kalau belum bisa dijalankan (mis. tanpa GPU/dataset), katakan apa adanya.

6. **Kode sederhana dan terbaca.** Saya mahasiswa yang harus menjelaskan tiap bagian saat sidang; utamakan kejelasan dibanding trik rumit. Beri komentar pada bagian penting (arsitektur multi-task, loss, preprocessing).

7. **Jangan ubah versi dependency besar** (framework, CUDA, dll.) tanpa konfirmasi, karena repo asli lara2018 relatif lama dan bisa konflik.

8. **Tanyakan kalau ambigu**, terutama soal skema label dan format ekspor.



## 8. Cara Berkomunikasi



- Saya berbicara dalam **Bahasa Indonesia informal**; balas dalam Bahasa Indonesia.

- Berikan jawaban yang **praktis, langsung, dan bisa dieksekusi**, bukan penjelasan generik.

- Istilah teknis (backbone, head, inference, dll.) boleh tetap dalam bahasa Inggris.



## 9. Roadmap Pengerjaan Model (urutan yang disarankan)



1. Audit repo lara2018: struktur, dependensi, alur training, bagian multi-task yang sudah ada.

2. Siapkan data pipeline untuk BRACOL, JMuBEN, JMuBEN v2 (loader + penyelarasan label + split bebas leakage). Cek ketersediaan label bbox; kalau tidak ada, usulkan dan minta persetujuan cara pembuatannya.

3. Implementasi model multi-task (shared backbone + head klasifikasi Softmax + head regresi keparahan + head lokalisasi bbox) dan baseline single-task, untuk backbone MobileNet dan EfficientNet (switch config `--backbone` dan `--mode`).

4. Training + evaluasi komparatif: 2 backbone x 2 skema (metrik, ukuran, kecepatan).

5. Pilih kombinasi terbaik (idealnya multi-task) berdasarkan trade-off akurasi vs ukuran/latensi.

6. Ekspor ke ONNX (utama, untuk Unity Sentis) dan TFLite, dengan output bernama sesuai kontrak 3.6. Verifikasi hasil ekspor sama dengan model asli.

7. Buat `model_meta.json`, skrip preprocessing+parsing yang meniru Unity, dan contoh output JSON untuk tim Unity.
apakah yang sudah kita kerjakan sudah sesuai dengan file .md ini? kalau blm tolong sesuaikan 
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-17T12:59:11+07:00.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Gemini 3.8 Flash (Medium) to Claude Opus 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>
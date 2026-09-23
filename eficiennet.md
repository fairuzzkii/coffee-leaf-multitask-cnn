# Task: Tambah Backbone EfficientNet ke Model Multi-Task (lara2018)

## Lokasi Project
```
C:\Users\fairu\Documents\fairuz\Penelitian Pak ranggi - Skripsi\lara2018-master
```
Semua eksplorasi, perubahan kode, dan training pada task ini dilakukan di
dalam folder project di atas.

## Konteks
Project ini adalah implementasi model multi-task deep learning untuk diagnosis
penyakit daun kopi, berdasarkan repo `esgario/lara2018`. Model memiliki 2 output
head:
1. **Head klasifikasi jenis penyakit** (disease classification)
2. **Head estimasi tingkat keparahan** (severity estimation)

Backbone yang sudah ada di project ini kemungkinan mencakup beberapa arsitektur
CNN (salah satunya **MobileNet**), yang masing-masing dipasangi 2 head yang sama
di atas. Tugas ini **tidak mengubah dataset atau training pipeline** — fokus
hanya pada penambahan opsi backbone baru.

## Tujuan
Tambahkan **EfficientNet (EfficientNet-B0)** sebagai opsi backbone baru,
sejajar dengan backbone MobileNet yang sudah ada, tanpa mengubah/merusak
backbone lain yang sudah berfungsi.

## Dataset & Training
**Jangan buat dataset loader baru.** Gunakan dataset dan cara loading data
yang **sudah ada di repo ini apa adanya** (folder dataset, script/notebook
training, konfigurasi path, dsb — apapun yang sudah dipakai untuk melatih
backbone MobileNet di project ini). Tujuannya supaya perbandingan MobileNet
vs EfficientNet adil (dataset, split train/val, augmentasi, dan cara training
persis sama, hanya backbone yang beda).

Langkah untuk bagian ini:
1. Cari script/notebook training yang sudah ada di repo (misal `train.py`,
   `main.py`, atau notebook `.ipynb`) yang dipakai untuk melatih model
   MobileNet.
2. Jalankan ulang training yang sama persis, tapi ganti parameter/argumen
   backbone-nya jadi `"efficientnet"` (setelah opsi ini ditambahkan sesuai
   langkah di bawah).
3. Kalau di repo ternyata sudah ada script perbandingan antar-backbone,
   pakai itu saja daripada membuat script baru.
4. Simpan hasil training EfficientNet (checkpoint + metric) dengan penamaan
   yang konsisten dengan hasil MobileNet yang sudah ada (misal folder/nama
   file mengandung nama backbone), supaya gampang dibandingkan.

## Langkah yang harus dilakukan

0. **Cek dulu apakah dataset gambar sudah ada di dalam folder project ini**
   (misal folder `dataset/`, `images/`, `data/` yang berisi file `.jpg`/`.png`
   asli, bukan cuma kode). Kalau folder project ini ternyata hanya berisi
   source code tanpa gambar dataset:
   - Cari di kode (README, script training, config) path dataset yang
     diharapkan.
   - Laporkan ke user path yang diharapkan tersebut dan folder mana yang
     kosong/belum ada, **jangan lanjut ke training** sebelum dataset
     tersedia di path yang benar. Penambahan backbone EfficientNet (langkah
     1-3) tetap bisa dilanjutkan meski dataset belum ada; hanya langkah
     training aktual yang perlu menunggu dataset siap.

1. **Eksplorasi struktur project terlebih dahulu.**
   Cari file yang mendefinisikan arsitektur model (biasanya bernama
   `model.py`, `models.py`, `network.py`, atau serupa). Temukan:
   - Fungsi/class yang memilih backbone berdasarkan nama/parameter
     (misal `build_model(architecture)` atau `get_model(arch_name)`).
   - Bagaimana 2 head (disease + severity) dipasang di atas fitur backbone.
   - Nama variabel/parameter yang dipakai untuk memilih backbone (misal
     string `"mobilenet"`, `"resnet"`, dst).

2. **Tambahkan opsi `"efficientnet"` mengikuti pola yang sama persis** dengan
   backbone lain yang sudah ada (terutama MobileNet), yaitu:
   - Gunakan `torchvision.models.efficientnet_b0` (pretrained ImageNet) sebagai
     feature extractor.
   - Hapus/ganti `classifier` bawaan EfficientNet dengan `nn.Identity()`
     supaya keluarannya berupa feature vector, bukan langsung logits.
   - Sambungkan feature vector tersebut ke 2 head yang **struktur dan
     dimensinya sama** dengan yang dipakai backbone lain di project ini
     (jangan buat head baru dari nol — ikuti pola/hyperparameter head yang
     sudah ada di kode, misal jumlah layer, dropout, dsb).

3. **Jangan ubah interface/signature fungsi yang sudah ada.**
   Model dengan backbone baru ini harus tetap bisa dipanggil dengan cara yang
   sama seperti backbone lain (misal `build_model("efficientnet", num_disease_classes, num_severity_classes)`),
   supaya training loop yang sudah ada tidak perlu diubah sama sekali.

4. **Tambahkan sanity check sederhana** (boleh berupa script kecil terpisah,
   misal `test_efficientnet_forward.py`) yang:
   - Membuat instance model dengan backbone `"efficientnet"`.
   - Menjalankan forward pass dengan dummy tensor `torch.randn(2, 3, 224, 224)`.
   - Mencetak/assert bahwa output berupa **dua tensor** dengan shape sesuai
     jumlah kelas disease dan severity yang diharapkan.
   - Pastikan tidak ada error saat load pretrained weights EfficientNet
     (butuh koneksi internet sekali saat pertama kali download weights).

5. **Jangan hapus atau ubah kode backbone lain** (MobileNet, ResNet, dst) yang
   sudah ada — tugas ini murni penambahan, bukan refactor besar.

## Kriteria selesai (Definition of Done)
- [ ] Ada opsi backbone baru `"efficientnet"` yang bisa dipilih di fungsi
      pembangun model, dengan struktur 2 head yang identik pola-nya dengan
      backbone lain di project ini.
- [ ] Forward pass dengan dummy input berhasil tanpa error, menghasilkan
      2 output tensor dengan shape yang benar.
- [ ] Backbone lain (terutama MobileNet) masih berfungsi seperti semula
      (tidak ada regresi).
- [ ] Laporkan file apa saja yang diubah/ditambahkan, dan tunjukkan potongan
      kode yang menambahkan opsi EfficientNet tersebut.
- [ ] Training EfficientNet berhasil dijalankan memakai dataset & script
      training yang sudah ada di repo (bukan dataset loader baru), dan
      hasilnya (metric, checkpoint) tersimpan dengan penamaan yang jelas
      untuk dibandingkan dengan hasil MobileNet.

## Catatan tambahan
- Dataset (BRACOL/JMuBEN) **tidak perlu disentuh** untuk task ini — cukup
  pastikan model bisa dibangun dan forward pass jalan dengan dummy data.
- Kalau ternyata pola backbone di project ini tidak berbentuk seperti dugaan
  di atas (misal 2 head tidak eksplisit terpisah, atau nama fungsinya beda),
  ikuti pola asli yang ditemukan di kode — jangan paksa mengikuti asumsi di
  brief ini kalau bertentangan dengan struktur kode yang sebenarnya.
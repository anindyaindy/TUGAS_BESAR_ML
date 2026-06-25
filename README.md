# Klasifikasi Citra Wajah Asli vs. Palsu (StyleGAN3) Menggunakan PyTorch

Proyek ini dibangun untuk mendeteksi manipulasi wajah (*deepfake*) dengan membandingkan wajah asli (REAL) dan wajah buatan AI (FAKE) berbasis arsitektur **EfficientNet-B3** dan **Xception** dengan framework **PyTorch**.

---

## 👥 Nama Anggota Kelompok

*   **Anindya Sukma Dwiyanda (anindyaindy)** - NIM: `103132400006`
*   **Jeremy Marcello (JeremyMarcello)** - NIM: `2311110003`

---

## 📝 Deskripsi Permasalahan
Perkembangan teknologi *Generative Adversarial Networks* (GAN), khususnya **StyleGAN3**, kini mampu menghasilkan citra wajah manusia buatan komputer dengan tingkat realisme yang sangat tinggi (*photorealistic*). Citra palsu ini sangat sulit dibedakan dari wajah asli oleh mata manusia biasa. Hal ini menimbulkan potensi ancaman keamanan berupa penyebaran disinformasi, penipuan identitas, maupun manipulasi media sosial. 

Oleh karena itu, proyek ini bertujuan untuk membangun sebuah sistem deteksi otomatis berbasis *Deep Learning* yang mampu mengklasifikasikan wajah secara akurat ke dalam kelas **REAL** (Asli) atau **FAKE** (Palsu buatan StyleGAN3).

---

## 📊 Sumber dan Deskripsi Dataset
*   **Sumber Dataset**: Dataset bersumber dari Kaggle: [Deepfake Detection Dataset 2026](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026) (juga tersedia lokal dalam folder `dataset/archive.zip` yang diekstrak menjadi `dataset/FINAL_DATASET.csv`).
*   **Deskripsi Data**:
    *   **Jumlah Baris**: 6.557 data gambar.
    *   **Distribusi Kelas**: 
        *   **REAL**: 2.790 gambar (bersumber dari Unsplash).
        *   **FAKE**: 3.767 gambar (dihasilkan melalui StyleGAN3).
    *   **Fitur Tabular/Metadata**: Kolom seperti `gender`, `age_group`, `image_quality`, `resolution`, `confidence_score`, `detection_difficulty`, dan `dataset_split`.
*   **Akses Gambar**: Citra diunduh langsung dari tautan URL pada kolom `image_url` menggunakan skrip pengunduh paralel.

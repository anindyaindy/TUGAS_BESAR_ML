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

---

## 🛠️ Tahapan Preprocessing
1.  **Validasi Citra**: Selama proses pengunduhan di `src/downloader.py`, setiap gambar yang berhasil diunduh dibuka menggunakan PIL `Image.open().convert('RGB')` untuk mendeteksi file korup dan disimpan seragam dalam format JPEG berkualitas tinggi (quality=92).
2.  **Transformasi & Augmentasi Training**:
    *   Mengubah resolusi gambar secara dinamis (300x300 piksel untuk EfficientNet-B3 dan 299x299 piksel untuk Xception).
    *   Penerapan augmentasi acak: `RandomHorizontalFlip` (p=0.5), `RandomRotation` (15 derajat), dan `ColorJitter` (brightness, contrast, saturation sebesar 15%) untuk mencegah model menghafal dataset.
    *   Normalisasi standar ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3.  **Transformasi Validasi & Pengujian**:
    *   Resize resolusi gambar sesuai model tanpa augmentasi acak.
    *   Normalisasi standar ImageNet.

---

## 🧠 Metode yang Digunakan
1.  **Arsitektur Model**:
    *   **EfficientNet-B3**: Model pretrained dengan arsitektur efisien berbasis MBConv (Mobile Inverted Bottleneck Convolution), diadaptasi dengan output biner logit.
    *   **Xception**: Model pretrained berbasis Depthwise Separable Convolution yang memiliki performa klasifikasi kuat.
2.  **Fungsi Loss (Loss Function)**:
    *   `BCEWithLogitsLoss` dengan perhitungan otomatis bobot kelas (`pos_weight`) untuk menangani ketidakseimbangan kelas (class imbalance) antara REAL (2.790 sampel) dan FAKE (3.767 sampel).
3.  **Optimizer**:
    *   `AdamW` dengan learning rate $10^{-4}$ dan dilengkapi regularisasi weight decay untuk membatasi kompleksitas bobot model.
4.  **Checkpointing**:
    *   Menyimpan model terbaik berdasarkan akurasi validasi tertinggi di setiap epoch untuk menghindari model yang mengalami overfitting pada epoch akhir.

---

## 🚀 Cara Menjalankan Program

### 1. Instalasi Dependensi
```bash
# Menginstal paket dependensi yang dibutuhkan secara langsung
pip install torch torchvision timm scikit-learn pandas matplotlib seaborn pillow tqdm
```

### 2. Jalankan Pengunduh Citra
```bash
python src/downloader.py --workers 8
```

### 3. Eksplorasi Data Analitis (EDA)
```bash
python eda.py
```

### 4. Pelatihan Model
*   **Melatih Model EfficientNet-B3 (Default)**:
    ```bash
    python train.py --model efficientnet_b3 --epochs 10 --batch_size 32 --lr 1e-4
    ```
*   **Melatih Model Xception**:
    ```bash
    python train.py --model xception --epochs 10 --batch_size 32 --lr 1e-4
    ```

### 5. Prediksi Gambar Tunggal (Inferensi)
```bash
python predict.py --image_path <path_ke_gambar.jpg> --model efficientnet_b3 --model_path checkpoints/best_efficientnet_b3.pth
```

---

## 📈 Hasil Eksperimen dan Evaluasi

Berikut adalah metrik performa final yang diperoleh setelah mengevaluasi model terbaik pada data pengujian (*Test Set*):

| Model | Loss Test | Akurasi Test | F1-Score Test | ROC-AUC Test |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B3** | 0.0044 | **99.81%** | **99.82%** | **1.0000** |
| **Xception** | 0.0001 | **100.00%** | **100.00%** | **1.0000** |

---

### 📊 Visualisasi Kurva Pelatihan & Confusion Matrix

#### A. EfficientNet-B3

##### Kurva Pelatihan
![Kurva Pelatihan EfficientNet-B3](visualisasi_plots/efficientnet_b3_training_history.png)
*   **Penjelasan Kurva**:
    *   **Loss Curve**: Baik *Train Loss* maupun *Validation Loss* menurun secara konsisten dari epoch pertama hingga epoch ke-10, mencapai nilai di bawah $0.005$. Penurunan paralel ini menandakan model belajar secara stabil tanpa mengalami overfitting (yang ditandai jika *Validation Loss* tiba-tiba naik kembali).
    *   **Accuracy Curve**: Akurasi pelatihan dan validasi meningkat cepat dan berhimpitan erat di kisaran $99.8\%$, membuktikan bahwa model memiliki kemampuan generalisasi yang sangat stabil untuk data baru.

##### Confusion Matrix
![Confusion Matrix EfficientNet-B3](visualisasi_plots/efficientnet_b3_confusion_matrix.png)
*   **Penjelasan Confusion Matrix**:
    *   Model EfficientNet-B3 memprediksi hampir seluruh citra dengan sempurna pada data uji (hanya terjadi 2 kesalahan klasifikasi dari 1.126 total citra pengujian). Ini menunjukkan tingkat kesalahan (*false positive* / *false negative*) yang sangat minimal.

---

#### B. Xception

##### Kurva Pelatihan
![Kurva Pelatihan Xception](visualisasi_plots/xception_training_history.png)
*   **Penjelasan Kurva**:
    *   **Loss Curve**: *Validation Loss* menurun dengan sangat mulus hingga mencapai nilai sangat kecil ($0.0001$). Kurva validasi berada di bawah kurva training, menunjukkan model sangat percaya diri dalam pengujian berkat generalisasi yang kuat dari fitur Xception.
    *   **Accuracy Curve**: Akurasi validasi mencapai $100\%$ sempurna sejak epoch ke-2 dan bertahan stabil, sementara akurasi training menyusul di epoch ke-8.

##### Confusion Matrix
![Confusion Matrix Xception](visualisasi_plots/xception_confusion_matrix.png)
*   **Penjelasan Confusion Matrix**:
    *   Matriks menunjukkan klasifikasi **$100\%$ sempurna** (nilai diagonal bernilai penuh, tidak ada gambar REAL yang terdeteksi FAKE maupun sebaliknya). Seluruh 1.126 citra uji berhasil diklasifikasikan dengan benar tanpa kesalahan sama sekali.

---

## 🎯 Kesimpulan
*   Kedua model yang dikembangkan memiliki performa luar biasa dalam mendeteksi manipulasi wajah (*deepfake*) StyleGAN3 dengan akurasi pengujian di atas 99.8%.
*   **Xception** unggul mutlak dengan akurasi 100.0% pada data uji, didukung oleh nilai loss validasi yang sangat kecil ($0.0001$).
*   **EfficientNet-B3** (ukuran file `43.3 MB`) jauh lebih ringan dibandingkan Xception (ukuran file `83.5 MB`). Sehingga, untuk kebutuhan deployment pada sistem dengan sumber daya terbatas (mobile/edge), **EfficientNet-B3** merupakan pilihan yang lebih efisien dengan penurunan performa yang tidak signifikan.

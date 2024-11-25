# Tugas-2-Kecerdasan-Artifisial-Muhammad Raza Adzani (2208107010066)
# Implementasi Jaringan Saraf Tiruan (Convolutional Neural Network)

Proyek ini bertujuan untuk mengimplementasikan algoritma **Jaringan Saraf Tiruan (JST)** berbasis **Convolutional Neural Network (CNN)** dalam menyelesaikan kasus **klasifikasi gambar bunga** menggunakan dataset "Flowers Recognition". Proyek ini menggunakan framework **TensorFlow** untuk perancangan dan pelatihan model.

---

## Pendahuluan

Jaringan Saraf Tiruan (JST) adalah salah satu teknik utama dalam kecerdasan buatan yang meniru cara kerja otak manusia. Pada proyek ini, JST digunakan untuk mengenali pola gambar bunga dan memprediksi kelasnya dengan memanfaatkan CNN sebagai arsitektur utama.

### Fitur Utama Proyek:
- Klasifikasi multi-kelas pada dataset gambar bunga.
- Arsitektur CNN dengan 7 hidden layers.
- Optimasi menggunakan Adam Optimizer.
- Aktivasi menggunakan ReLU (hidden layers) dan Softmax (output layer).

---

## Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python
- **Framework:** TensorFlow
- **Library Tambahan:** Matplotlib, NumPy, Pandas
- **Dataset:** [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

---

## Langkah-Langkah Implementasi

1. **Persiapan Dataset**
   - Dataset: Flowers Recognition
   - Dimensi Gambar: 150x150x3 piksel
   - Total Kelas: 5 kelas (Daisy, Dandelion, Rose, Sunflower, Tulip)

2. **Arsitektur Model CNN**
   - 3 Hidden Layer Conv2D
   - 3 Hidden Layer BatchNormalization
   - 1 Dense Layer
   - Total Trainable Parameters: **319,717**

3. **Proses Training**
   - Optimizer: Adam
   - Fungsi Aktivasi: ReLU (hidden layers), Softmax (output layer)
   - Plot proses training disimpan di `results/training_plot.png`.

4. **Evaluasi**
   - Akurasi model dicatat dengan TensorBoard dan hasil evaluasi model disimpan di `results/accuracy_screenshot.png`.

---

## Hasil dan Kesimpulan

Model CNN yang dirancang mampu melakukan klasifikasi gambar bunga dengan kompleksitas parameter yang cukup untuk mempelajari pola data. Hasilnya mencerminkan potensi JST sebagai alat yang efektif dalam menangani masalah klasifikasi gambar berbasis data non-linear.

---



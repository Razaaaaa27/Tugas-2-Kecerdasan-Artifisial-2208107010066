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

##  Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python
- **Framework:** TensorFlow, Keras
- **Library Tambahan:** Matplotlib, NumPy, Pandas
- **Dataset:** [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

---

##  Struktur Proyek

```plaintext
📂 Tugas-2-Kecerdasan-Artifisial-NPM
├── 📁 source_code
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── utils.py
├── 📁 saved_model
│   └── model.h5
├── 📁 results
│   ├── training_plot.png
│   ├── accuracy_screenshot.png
│   ├── tensorboard_logs/
├── 📁 reports
│   ├── laporan_proyek.pptx
├── README.md

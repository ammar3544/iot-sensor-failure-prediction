# Prediksi Kegagalan Sensor IoT (Handling Imbalance & Data Leakage)

Repositori ini mengimplementasikan studi kasus analitik data dan AI untuk sistem IoT, berfokus pada pencegahan *data leakage* dan penanganan *class imbalance* pada data sensor[cite: 2].

## Fitur Proyek
- **Eliminasi Data Leakage**: Membuang variabel pasca-kejadian (`maintenance_downtime_hours`).
- **Penanganan Class Imbalance**: Perbandingan performa antara Baseline, `Class Weight`, dan `SMOTE`.
- **Metrik Evaluasi Berbasis Recall**: Memastikan model benar-benar mendeteksi kelas minoritas (*Failure*).

## Cara Menjalankan
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan skrip: `python main.py`
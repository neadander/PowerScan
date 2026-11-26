# PowerScan — Deteksi Lonjakan Listrik (Streamlit)

Aplikasi web untuk deteksi anomali/lonjakan konsumsi listrik per rumah menggunakan
Isolation Forest, port dari pipeline notebook `PowerScan_IsolationForest.ipynb`.

## Fitur

- Unggah CSV data per-jam (mendukung format Indonesia: separator `;`, desimal koma)
- Preprocessing & feature engineering identik dengan notebook (8 fitur, rolling 24 jam)
- Model Isolation Forest terpisah per rumah (`MinMaxScaler` per rumah)
- Ringkasan jumlah lonjakan, grafik interaktif Plotly, analisis penyebab + saran
- Download hasil CSV

## Kolom CSV yang dibutuhkan

Wajib: `House_ID`, `Date`, `Time`, `Global_active_power`, `Voltage`, `Global_intensity`.
Opsional: `Global_reactive_power`, `Sub_metering_1/2/3`, `Is_Anomaly`, dst.

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru (mis. `powerscan`) dan upload file:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `powerscan_app.py` tidak perlu diupload (duplikat `app.py`)
2. Buka https://share.streamlit.io → **New app**
3. Pilih repository, branch `main`, **Main file path**: `app.py`
4. Klik **Deploy** — selesai, aplikasi dapat diakses via URL publik.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Fungsi untuk memuat model dan scaler, di-cache agar lebih cepat
@st.cache_resource
def load_model_and_scaler():
    # Memuat model dengan format .keras yang baru dan lebih modern
    model = tf.keras.models.load_model("model_wti.keras", compile=False, safe_mode=False)

    # Memuat scaler
    scaler_y = joblib.load("scaler_y_case_1.pkl")

    return model, scaler_y

# Fungsi untuk memuat dan membersihkan data, di-cache agar lebih cepat
@st.cache_data
def load_data(filepath):
    """Memuat data dari CSV, membersihkan kolom tanggal."""
    data = pd.read_csv(filepath)
    # 1. Ubah ke datetime, paksa nilai error menjadi NaT (Not a Time)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    
    # 2. Hapus semua baris di mana kolom 'Date' memiliki nilai NaT
    data.dropna(subset=['Date'], inplace=True)
    
    return data

# --- [INI ADALAH FUNGSI YANG DIPERBAIKI TOTAL] ---
def forecast_future(model, initial_sequence, horizon):
    """
    Melakukan prediksi iteratif untuk beberapa hari ke depan.
    
    Args:
        model: Model Keras yang sudah di-train.
        initial_sequence: 30 data terakhir sebagai input awal.
        horizon: Jumlah hari yang akan diprediksi.
        
    Returns:
        NumPy array berisi nilai-nilai prediksi.
    """
    # Salin sekuens awal agar tidak mengubah data asli
    current_sequence = list(initial_sequence.flatten())
    
    # List ini akan menampung hasil prediksi
    future_predictions = []

    for _ in range(horizon):
        # 1. Siapkan input untuk model
        input_for_pred = np.array(current_sequence).reshape(1, len(current_sequence), 1)
        
        # 2. Dapatkan prediksi dari model (output-nya adalah NumPy array)
        prediction_scaled = model.predict(input_for_pred, verbose=0)
        
        # 3. Ambil nilai tunggal (float) dari hasil prediksi
        # PERBAIKAN: Ambil elemen pertama [0] jika outputnya berupa list
        next_value = prediction_scaled[0].flatten()[0]
        
        # 4. Simpan nilai tunggal tersebut ke dalam list hasil
        future_predictions.append(next_value)
        
        # 5. Perbarui sekuens input untuk iterasi berikutnya:
        #    - Hapus data tertua
        current_sequence.pop(0)
        #    - Tambahkan data prediksi terbaru
        current_sequence.append(next_value)
        
    # Kembalikan hasil prediksi sebagai NumPy array
    return np.array(future_predictions)

# --- [UI STREAMLIT] ---

st.title("📈 Forecasting Harga Minyak Mentah WTI")
st.write("Model ini menggunakan **30 data historis WTI** untuk memprediksi harga pada hari-hari berikutnya.")

# Memuat model dan data dengan penanganan error
try:
    model, scaler_y = load_model_and_scaler()
    data = load_data("Data PA Fix_revisi.csv")
except Exception as e:
    st.error(f"Error: Gagal memuat file model, scaler, atau data. Pastikan file ada dan tidak rusak.")
    st.error(f"Detail error: {e}")
    st.info("Pastikan file `best_model_case_1.h5`, `scaler_y_case_1.pkl`, dan `Data PA Fix_revisi.csv` berada di folder yang sama dengan `app.py`.")
    st.stop()

# Menyiapkan data untuk kalkulasi MAPE dan prediksi
y_raw = data["WTI"].values.reshape(-1, 1)
y_scaled = scaler_y.transform(y_raw)

# --- [BAGIAN BARU: Kalkulasi dan Tampilkan MAPE] ---
st.subheader("📊 Performa Model")
time_steps = 30
test_period = 90 # Menggunakan 90 hari terakhir sebagai data uji

if len(y_scaled) > test_period + time_steps:
    # Menyiapkan data uji
    X_test_sequences = []
    y_test_sequences = []
    data_for_test = y_scaled[-(test_period + time_steps):]

    for i in range(len(data_for_test) - time_steps):
        X_test_sequences.append(data_for_test[i:(i + time_steps), 0])
        y_test_sequences.append(data_for_test[i + time_steps, 0])
    
    X_test = np.array(X_test_sequences)
    y_test = np.array(y_test_sequences).reshape(-1, 1)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Membuat prediksi pada data uji
    test_pred_scaled = model.predict(X_test, verbose=0)
    
    # --- [INI ADALAH BARIS YANG DIPERBAIKI] ---
    # Ekstrak array prediksi dari list output model [0] sebelum inverse transform
    test_pred_unscaled = scaler_y.inverse_transform(test_pred_scaled[0])
    
    y_test_unscaled = scaler_y.inverse_transform(y_test)
    
    # Hitung dan tampilkan MAPE
    mape = mean_absolute_percentage_error(y_test_unscaled, test_pred_unscaled) * 100
    st.metric("MAPE pada Data Uji (90 hari terakhir)", f"{mape:.2f} %")
else:
    st.warning("Data tidak cukup untuk menghitung MAPE pada data uji.")
# ----------------------------------------------------

st.subheader("🔮 Buat Prediksi Baru")
# Slider untuk memilih horizon
horizon = st.slider("Pilih horizon forecast (hari):", min_value=1, max_value=90, value=30, step=1)

# Menyiapkan data terakhir untuk prediksi masa depan
last_30_days_scaled = y_scaled[-30:]

if st.button("Mulai Forecast"):
    with st.spinner("Memproses prediksi... Mohon tunggu sebentar."):
        # Panggil fungsi forecast yang sudah benar
        pred_scaled = forecast_future(model, last_30_days_scaled, horizon)
        
        # Kembalikan prediksi ke skala asli
        pred_unscaled = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

        # --- [BAGIAN DIREVISI: Plotting disederhanakan] ---
        # Buat plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Siapkan tanggal untuk masa depan
        last_date = data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        
        # Plot HANYA data forecast
        ax.plot(future_dates, pred_unscaled, label="Hasil Forecast", color="red", marker='o', linestyle='--')
        
        # Styling plot
        ax.set_title("Hasil Forecast Harga WTI")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        # --------------------------------------------------

        # Tampilkan detail dalam tabel
        st.subheader("Detail Hasil Forecast:")
        forecast_df = pd.DataFrame({
            'Tanggal': future_dates,
            'Prediksi Harga WTI (USD)': pred_unscaled.flatten()
        })
        
        # Format tabel agar lebih mudah dibaca
        forecast_df['Tanggal'] = forecast_df['Tanggal'].dt.strftime('%d-%m-%Y')
        forecast_df['Prediksi Harga WTI (USD)'] = forecast_df['Prediksi Harga WTI (USD)'].map('${:,.2f}'.format)
        
        st.dataframe(forecast_df, use_container_width=True)


        st.success("✅ Proses forecasting selesai!")

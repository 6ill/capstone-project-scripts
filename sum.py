import pandas as pd

# Menunjuk ke lokasi metadata.csv di dalam folder dataset_final
nama_file = 'dataset_final/metadata.csv'

try:
    # Membaca file CSV
    df = pd.read_csv(nama_file)

    # Menghitung total durasi dari kolom 'duration' (dalam detik)
    total_duration = df['duration'].sum()

    # Menghitung konversi ke jam, menit, dan sisa detik
    jam = int(total_duration // 3600)
    menit = int((total_duration % 3600) // 60)
    detik = total_duration % 60

    # Menampilkan hasil
    print(f"Total durasi keseluruhan adalah: {total_duration:.2f} detik")
    print(f"Atau sama dengan: {jam} jam, {menit} menit, dan {detik:.2f} detik")

except FileNotFoundError:
    print(f"Error: File '{nama_file}' tidak ditemukan. Pastikan kamu menjalankan script dari folder SCRIPTS.")
except KeyError:
    print("Error: Kolom 'duration' tidak ditemukan di dalam file CSV.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
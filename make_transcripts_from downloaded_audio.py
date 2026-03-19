import os
import sys
import glob
import time
import gc # [BARU] Untuk membersihkan memori sistem

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import pandas as pd
from faster_whisper import WhisperModel
from pydub import AudioSegment
import torch
from tqdm import tqdm

# --- KONFIGURASI ---
DATASET_DIR = "dataset_final"
RAW_DIR = "temp_raw"
HISTORY_FILE = "download_history.txt"

# Path folder tempat menyimpan video/audio input
SOURCE_FOLDER = "F:/TABillGraceHizkia/scripts/dataset"  # Ganti dengan path folder Anda
MAX_FILES = 120

# Konfigurasi jeda (dalam milidetik)
SLEEP_MS = 1000 # 1000 ms = 1 detik

VALID_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mp3', '.wav', '.m4a', '.flac')

SAMPLE_RATE = 16000
MODEL_SIZE = "large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

os.makedirs(f"{DATASET_DIR}/wavs", exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def is_processed(file_id):
    if not os.path.exists(HISTORY_FILE): return False
    with open(HISTORY_FILE, 'r') as f: return file_id in f.read().splitlines()

def mark_as_processed(file_id):
    with open(HISTORY_FILE, 'a') as f: f.write(f"{file_id}\n")

# [PERBAIKAN] Tambahkan parameter 'model' ke dalam fungsi ini
def process_local_file(file_path, model):
    filename_with_ext = os.path.basename(file_path)
    file_id = os.path.splitext(filename_with_ext)[0]
    
    print(f"\n🚀 Memproses: {filename_with_ext}")

    if is_processed(file_id):
        print("   [SKIP] Sudah diproses sebelumnya (ada di history).")
        return False

    audio_temp_path = f"{RAW_DIR}/{file_id}.wav"
    
    print("   [Converting] Mengubah format ke WAV 16kHz Mono...")
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(audio_temp_path, format="wav")
    except Exception as e:
        print(f"   [FAIL] Gagal konversi audio: {e}")
        return False

    print(f"   [Processing] Transkripsi dengan model {MODEL_SIZE} ({DEVICE})...")
    
    try:
        # [PERBAIKAN] Model TIDAK LAGI di-load di sini, langsung pakai yang dikirim dari luar
        segments, info = model.transcribe(
            audio_temp_path, 
            vad_filter=True, 
            language="id", 
            vad_parameters=dict(min_silence_duration_ms=1500)
        )

        source_audio = AudioSegment.from_wav(audio_temp_path)
        
        new_entries = []
        seg_count = 0

        print(f"   [Info] Total Durasi: {info.duration:.2f} detik")
        with tqdm(total=round(info.duration), unit="detik", desc="   Proses", ncols=80) as pbar:
            for segment in segments:
                start_s = segment.start
                end_s = segment.end
                text = segment.text.strip()
                duration = end_s - start_s
                
                pbar.update(round(end_s) - pbar.n)

                if duration < 2.0 or duration > 25.0:
                    continue

                start_ms = start_s * 1000
                end_ms = end_s * 1000
                chunk = source_audio[start_ms:end_ms]
                
                out_filename = f"{file_id}_{seg_count}.wav"
                chunk.export(f"{DATASET_DIR}/wavs/{out_filename}", format="wav")
                
                new_entries.append({
                    "path": f"wavs/{out_filename}",
                    "text": text,
                    "duration": round(duration, 2),
                    "original_source": filename_with_ext
                })
                seg_count += 1

        if new_entries:
            df = pd.DataFrame(new_entries)
            csv_path = f"{DATASET_DIR}/metadata.csv"
            header = not os.path.exists(csv_path)
            df.to_csv(csv_path, mode='a', index=False, header=header)
            
            mark_as_processed(file_id)
            print(f"   ✅ Sukses! {len(new_entries)} segmen tersimpan.")
            result_status = True
        else:
            print("   ⚠️ Tidak ada segmen valid.")
            mark_as_processed(file_id) 
            result_status = True

    except Exception as e:
        print(f"   [ERROR] Transkripsi Gagal: {e}")
        import traceback
        traceback.print_exc()
        result_status = False

    if os.path.exists(audio_temp_path): os.remove(audio_temp_path)
    
    return result_status

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ Error: Folder '{SOURCE_FOLDER}' tidak ditemukan!")
        exit()

    all_files = []
    print(f"📂 Scanning folder: {SOURCE_FOLDER} ...")
    
    for filename in os.listdir(SOURCE_FOLDER):
        if filename.lower().endswith(VALID_EXTENSIONS):
            all_files.append(os.path.join(SOURCE_FOLDER, filename))
    
    print(f"   Ditemukan {len(all_files)} file media.")
    print(f"🎯 Target proses kali ini: {MAX_FILES} file.")

    processed_count = 0

    # [PERBAIKAN UTAMA] Load model HANYA 1x di sini!
    print(f"\n🧠 Memuat model Whisper {MODEL_SIZE} ke VRAM (Hanya 1x)...")
    global_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    for file_path in all_files:
        if processed_count >= MAX_FILES:
            print(f"\n🛑 Berhenti: Mencapai batas limit {MAX_FILES} file.")
            break

        # Oper model yang sudah di-load ke dalam fungsi
        is_success = process_local_file(file_path, global_model)
        
        if is_success:
            processed_count += 1
            
        # [PERBAIKAN] Bersihkan memori GPU & RAM agar tidak menumpuk!
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"   ⏳ Jeda {SLEEP_MS}ms agar GPU tidak overheat...")
        time.sleep(SLEEP_MS / 1000.0) 

    print("\n✅ Semua tugas selesai.")
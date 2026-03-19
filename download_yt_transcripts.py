import os

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import yt_dlp
import pandas as pd
from faster_whisper import WhisperModel
from pydub import AudioSegment
import torch
from tqdm import tqdm


# --- KONFIGURASI ---
DATASET_DIR = "dataset_final"
RAW_DIR = "temp_raw"
HISTORY_FILE = "download_history.txt"
SAMPLE_RATE = 16000

# Model Size: 'medium' atau 'large-v2' (Sangat akurat untuk Indonesia)
MODEL_SIZE = "large-v2"
# Gunakan 'cuda' jika ada GPU NVIDIA, jika tidak 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

os.makedirs(f"{DATASET_DIR}/wavs", exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def is_processed(video_id):
    if not os.path.exists(HISTORY_FILE):
        return False
    with open(HISTORY_FILE, "r") as f:
        return video_id in f.read().splitlines()


def mark_as_processed(video_id):
    with open(HISTORY_FILE, "a") as f:
        f.write(f"{video_id}\n")


def process_stable(video_url):
    print(f"\n🚀 Memproses: {video_url}")

    # 1. Download Audio
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{RAW_DIR}/%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "impersonate_target": "chrome-110:windows-10",
    }

    video_id = ""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info["id"]
            title = info["title"]
    except Exception as e:
        print(f"[FAIL] Download Error: {e}")
        return

    if is_processed(video_id):
        print("[SKIP] Sudah diproses.")
        return

    audio_path = f"{RAW_DIR}/{video_id}.wav"

    # 2. Transcribe dengan Faster-Whisper
    # vad_filter=True adalah pengganti VAD di WhisperX (membuang bagian hening)
    print(f"   [Processing] Transkripsi dengan model {MODEL_SIZE} ({DEVICE})...")

    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

        # vad_filter=True -> Otomatis potong bagian hening
        # min_silence_duration_ms -> Durasi hening minimal untuk dianggap jeda
        segments, info = model.transcribe(
            audio_path,
            vad_filter=True,
            language="id",
            vad_parameters=dict(
                min_silence_duration_ms=1500,
            ),
        )

        # 3. Slicing Audio
        source_audio = AudioSegment.from_wav(audio_path)
        source_audio = source_audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

        new_entries = []
        seg_count = 0
        
        print(f"Total durasi: {info.duration} detik")
        with tqdm(total=round(info.duration), unit="detik", desc="   Proses", ncols=80) as pbar:
            for segment in segments:
                start_s = segment.start
                end_s = segment.end
                text = segment.text.strip()
                duration = end_s - start_s
                
                pbar.update(round(end_s) - pbar.n)

                # Filter Durasi (2 detik s/d 20 detik)
                if duration < 2.0 or duration > 25.0:
                    continue

                # Potong
                start_ms = start_s * 1000
                end_ms = end_s * 1000
                chunk = source_audio[start_ms:end_ms]

                filename = f"{video_id}_{seg_count}.wav"
                chunk.export(f"{DATASET_DIR}/wavs/{filename}", format="wav")

                new_entries.append(
                    {
                        "path": f"wavs/{filename}",
                        "text": text,
                        "duration": duration,
                        "original_video_id": video_id,
                    }
                )
                seg_count += 1

        # Save CSV
        if new_entries:
            df = pd.DataFrame(new_entries)
            csv_path = f"{DATASET_DIR}/metadata.csv"
            header = not os.path.exists(csv_path)
            df.to_csv(csv_path, mode="a", index=False, header=header)
            mark_as_processed(video_id)
            print(f"   ✅ Sukses! {len(new_entries)} data tersimpan.")
        else:
            print("   ⚠️ Tidak ada segmen valid.")

    except Exception as e:
        print(f"   [ERROR] Transkripsi Gagal: {e}")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)


if __name__ == "__main__":
    urls = ["https://www.youtube.com/watch?v=hQNixcmbTNo"]
    for url in urls:
        process_stable(url)

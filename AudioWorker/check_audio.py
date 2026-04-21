import soundfile as sf

audio_path = "audio/meeting_clean.wav"

data, samplerate = sf.read(audio_path)

duration_seconds = len(data) / samplerate
duration_minutes = duration_seconds / 60

print("Ses başariyla yüklendi ✅")
print(f"Örnekleme hizi: {samplerate} Hz")
print(f"Toplam süre: {duration_seconds:.2f} saniye")
print(f"Toplam süre: {duration_minutes:.2f} dakika")
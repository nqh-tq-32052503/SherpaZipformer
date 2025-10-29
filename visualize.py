import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

class Visualization(object):
    def __init__(self):
        self.modes = ["waveform", "spectrogram", "mel", "mfcc", "pitch", "rms-energy"]
        print("[INFO] Available modes: ", self.modes)
    
    def plot(self, filepath, mode):
        assert os.path.exists(filepath), "[ERROR] File not found"
        assert mode in self.modes, "[ERROR] Mode is not available"
        if mode == "waveform":
            self.plot_waveform(filepath)
        elif mode == "spectrogram":
            self.plot_spectrogram(filepath)
        elif mode == "mel":
            self.plot_mel_spectrogram(filepath)
        elif mode == "mfcc":
            self.plot_mfcc(filepath)
        elif mode == "pitch":
            self.plot_pitch(filepath)
        elif mode == "rms-energy":
            self.plot_rms_energy(filepath)

    def plot_waveform(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        plt.figure(figsize=(10, 3))
        plt.plot(y, linewidth=0.5)
        plt.title("Waveform")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        S = np.abs(librosa.stft(y))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram (dB)")
        plt.tight_layout()
        plt.show()

    def plot_mel_spectrogram(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel-Spectrogram")
        plt.tight_layout()
        plt.show()
    
    def plot_mfcc(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title("MFCCs")
        plt.tight_layout()
        plt.show()

    def plot_pitch(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500)
        times = librosa.times_like(f0)
        plt.figure(figsize=(10, 3))
        plt.plot(times, f0, label='F0', linewidth=1)
        plt.title("Pitch Contour (Fundamental Frequency)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()
    
    def plot_rms_energy(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        frames = range(len(rms))
        times = librosa.frames_to_time(frames, sr=sr)
        plt.figure(figsize=(10, 3))
        plt.plot(times, rms, linewidth=1)
        plt.title("RMS Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.tight_layout()
        plt.show()

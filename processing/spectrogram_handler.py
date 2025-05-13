import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from config import *

class SpectrogramHandler:
    def __init__(self, save_enabled, channels):
        self.save_enabled = save_enabled
        self.channels = channels
        self.counter = 0
        self.log = []

    def process(self, fragment, timestamp_seconds):
        if not self.save_enabled:
            return

        for ch in range(self.channels):
            mel_spec = librosa.feature.melspectrogram(
                y=fragment[:, ch],
                sr=samplerate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            filename = f"spectrogram_{self.counter:05d}_ch{ch}.png"
            filepath = os.path.join(output_folder, filename)

            x_coords = np.arange(mel_spec_db.shape[1]) * hop_length / samplerate + timestamp_seconds

            plt.figure(figsize=(8, 4))
            librosa.display.specshow(
                mel_spec_db,
                sr=samplerate,
                hop_length=hop_length,
                x_axis='time',
                y_axis='mel',
                x_coords=x_coords
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram {self.counter} Channel {ch}')
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()

            self.log.append([filename, timestamp_seconds])

        self.counter += 1

    def save_log(self):
        if not self.save_enabled:
            return

        csv_path = os.path.join(output_folder, 'spectrogram_log.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['spectrogram_filename', 'timestamp_seconds'])
            writer.writerows(self.log)
        print(f"Log de espectrogramas guardado en {csv_path}")
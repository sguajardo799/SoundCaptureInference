# processing/inference_handler.py

import numpy as np
import csv
import os
from config import *

class InferenceHandler:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.predictions_log = []

    def process(self, fragment, timestamp_seconds):
        for ch in range(fragment.shape[1]):
            mel_spec = librosa.feature.melspectrogram(
                y=fragment[:, ch],
                sr=samplerate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            input_tensor = mel_spec_db[np.newaxis, np.newaxis, :, :].astype(np.float32)
            prediction = self.model_runner.predict(input_tensor)
            self.predictions_log.append([timestamp_seconds, ch, prediction.tolist()])

    def save_log(self):
        pred_path = os.path.join(output_folder, 'predictions_log.csv')
        with open(pred_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_seconds', 'channel', 'prediction'])
            writer.writerows(self.predictions_log)
        print(f"Log de inferencias guardado en {pred_path}")

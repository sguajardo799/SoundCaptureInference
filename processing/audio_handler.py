# processing/audio_handler.py

import numpy as np
import os
import soundfile as sf
import queue
import shutil
import threading
from config import *

class AudioHandler:
    def __init__(self, channels):
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.buffer_samples = int(max_buffer_seconds * samplerate)
        self.audio_buffer = np.zeros((self.buffer_samples, channels), dtype=np.float32)
        self.write_index = 0
        self.samples_processed = 0
        self.saved_audio_chunks = []
        self.audio_queue = queue.Queue()
        self.recording = True

        self.global_time_seconds = 0.0  # <<< NUEVO: contador de tiempo real

        os.makedirs(temp_folder, exist_ok=True)

    def feed(self, indata, frames):
        for i in range(frames):
            self.audio_buffer[self.write_index, :] = indata[i, :]
            self.write_index = (self.write_index + 1) % self.buffer_samples
        self.audio_queue.put(frames)

    def stop(self):
        self.recording = False

    def process_audio(self, spectrogram_handler, inference_handler):
        samples_per_interval = int(spectrogram_interval * samplerate)
        chunk_counter = 0

        while self.recording or not self.audio_queue.empty():
            try:
                samples_received = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.samples_processed += samples_received

            while self.samples_processed >= samples_per_interval:
                start_idx = (self.write_index - self.samples_processed) % self.buffer_samples
                end_idx = (start_idx + samples_per_interval) % self.buffer_samples

                if start_idx < end_idx:
                    fragment = self.audio_buffer[start_idx:end_idx, :]
                else:
                    fragment = np.vstack((
                        self.audio_buffer[start_idx:],
                        self.audio_buffer[:end_idx]
                    ))

                timestamp_seconds = self.global_time_seconds  # <<< USAMOS TIEMPO GLOBAL

                if spectrogram_handler:
                    spectrogram_handler.process(fragment, timestamp_seconds)

                if inference_handler:
                    inference_handler.process(fragment, timestamp_seconds)

                self.samples_processed -= samples_per_interval
                self.global_time_seconds += spectrogram_interval  # <<< AVANZAMOS el tiempo real

            if self.samples_processed >= self.buffer_samples:
                print(f"Volcando buffer a disco: fragmento {chunk_counter}")
                chunk_filename = f"buffer_chunk_{chunk_counter:05d}.wav"
                chunk_path = os.path.join(temp_folder, chunk_filename)
                chunk_data = np.vstack((
                    self.audio_buffer[(self.write_index-self.buffer_samples):],
                    self.audio_buffer[:self.write_index]
                ))
                sf.write(chunk_path, chunk_data, samplerate)
                self.saved_audio_chunks.append(chunk_path)
                chunk_counter += 1
                self.samples_processed = 0

    def save_all(self):
        print("Guardando audio completo...")
        output_path = os.path.join(output_folder, 'audio_recorded.wav')
        audio_pieces = []

        for chunk_path in self.saved_audio_chunks:
            audio, _ = sf.read(chunk_path, dtype='float32')
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]
            audio_pieces.append(audio)

        if self.write_index > 0:
            final_audio = self.audio_buffer[:self.write_index, :]
            audio_pieces.append(final_audio)

        if audio_pieces:
            full_audio = np.vstack(audio_pieces)
            sf.write(output_path, full_audio, samplerate)
            print(f"Audio guardado en {output_path}")

        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Carpeta temporal {temp_folder} eliminada.")

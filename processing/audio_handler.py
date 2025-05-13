# processing/audio_handler.py

import numpy as np
import os
import soundfile as sf
import queue
import shutil
from config import *

class AudioHandler:
    def __init__(self, channels, step_duration_seconds=None):
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.buffer_samples = int(max_buffer_seconds * samplerate)
        self.audio_buffer = np.zeros((self.buffer_samples, channels), dtype=np.float32)
        self.write_index = 0
        self.read_index = 0  # índice absoluto de lectura
        self.samples_processed = 0
        self.saved_audio_chunks = []
        self.audio_queue = queue.Queue()
        self.recording = True

        self.window_samples = int(spectrogram_interval * samplerate)
        self.step_samples = int(step_duration_seconds * samplerate) if step_duration_seconds else self.window_samples
        self.step_duration_seconds = self.step_samples / samplerate

        self.total_audio = []  # para almacenar bloques de audio

        os.makedirs(temp_folder, exist_ok=True)

    def feed(self, indata, frames):
        for i in range(frames):
            self.audio_buffer[self.write_index, :] = indata[i, :]
            self.write_index = (self.write_index + 1) % self.buffer_samples
        self.audio_queue.put(indata.copy())

    def stop(self):
        self.recording = False

    def process_audio(self, spectrogram_handler, inference_handler):
        chunk_counter = 0
        absolute_write_index = 0

        while self.recording or not self.audio_queue.empty():
            try:
                block = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            num_samples = block.shape[0]
            self.samples_processed += num_samples
            self.total_audio.append(block)
            absolute_write_index += num_samples

            while (self.read_index + self.window_samples) <= absolute_write_index:
                start_idx = self.read_index % self.buffer_samples
                end_idx = (start_idx + self.window_samples) % self.buffer_samples

                if start_idx < end_idx:
                    fragment = self.audio_buffer[start_idx:end_idx, :]
                else:
                    fragment = np.vstack((
                        self.audio_buffer[start_idx:],
                        self.audio_buffer[:end_idx]
                    ))

                timestamp_seconds = self.read_index / samplerate

                if spectrogram_handler:
                    spectrogram_handler.process(fragment, timestamp_seconds)
                if inference_handler:
                    inference_handler.process(fragment, timestamp_seconds)

                self.read_index += self.step_samples

        # Volcado seguro solo si se grabó algo
        if self.total_audio:
            print("Volcando audio total grabado...")
            chunk_filename = f"buffer_chunk_{chunk_counter:05d}.wav"
            chunk_path = os.path.join(temp_folder, chunk_filename)
            audio_data = np.vstack(self.total_audio)
            sf.write(chunk_path, audio_data, samplerate)
            self.saved_audio_chunks.append(chunk_path)

    def save_all(self):
        print("Guardando audio completo...")
        output_path = os.path.join(output_folder, 'audio_recorded.wav')
        audio_pieces = []

        for chunk_path in self.saved_audio_chunks:
            if os.path.exists(chunk_path):
                audio, _ = sf.read(chunk_path, dtype='float32')
                if audio.ndim == 1:
                    audio = audio[:, np.newaxis]
                audio_pieces.append(audio)

        if audio_pieces:
            full_audio = np.vstack(audio_pieces)
            sf.write(output_path, full_audio, samplerate)
            print(f"Audio guardado en {output_path}")

        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Carpeta temporal {temp_folder} eliminada.")

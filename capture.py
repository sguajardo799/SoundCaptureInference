import argparse
from processing.audio_handler import AudioHandler
from processing.spectrogram_handler import SpectrogramHandler
from processing.inference_handler import InferenceHandler
from model_runner.base import ModelRunner
import sounddevice as sd
import time
import threading

def parse_arguments():
    parser = argparse.ArgumentParser(description="Grabador de audio con inferencia opcional.")
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--save_spectrograms', action='store_true')
    parser.add_argument('--use_inference', action='store_true')
    parser.add_argument('--ml_model_path', type=str, default=None)
    parser.add_argument('--ml_framework', type=str, choices=['onnx', 'pytorch', 'hailo'], default='onnx')
    return parser.parse_args()

def main():
    args = parse_arguments()

    audio_handler = AudioHandler(channels=args.channels, step_duration_seconds=2)
    spectrogram_handler = SpectrogramHandler(save_enabled=args.save_spectrograms, channels=args.channels)
    inference_handler = None

    if args.use_inference:
        model = ModelRunner(args.ml_model_path, args.ml_framework)
        inference_handler = InferenceHandler(model_runner=model)

    def audio_callback(indata, frames, time_info, status):
        audio_handler.feed(indata, frames)

    processing_thread = threading.Thread(
        target=audio_handler.process_audio,
        args=(spectrogram_handler, inference_handler),
        daemon=True
    )
    processing_thread.start()

    with sd.InputStream(
        samplerate=audio_handler.samplerate,
        channels=args.channels,
        blocksize=audio_handler.blocksize,
        dtype='float32',
        callback=audio_callback
    ):
        print(f"Grabando {'estéreo' if args.channels == 2 else 'mono'}... Ctrl+C para detener.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print('Finalizando grabación...')

    audio_handler.stop()
    processing_thread.join()  # Espera a que termine el hilo de procesamiento

    audio_handler.save_all()
    spectrogram_handler.save_log()
    if inference_handler:
        inference_handler.save_log()

if __name__ == "__main__":
    main()

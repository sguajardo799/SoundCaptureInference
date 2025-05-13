# Sound Capture Project

Captura de audio en tiempo real, generación de espectrogramas de Mel y aplicación de inferencias de ML.

## Dependencias Externas
Para Linux instalar libportaudio2 y libasound2-dev

## Estructura
- `capture.py` — Main
- `processing/` — Audio, Spectrogram, Inference
- `model_runner/` — Model wrapper
- `config.py` — Parámetros

## Ejemplo de uso

```bash
python capture.py --channels 2 --save_spectrograms --use_inference --ml_model_path model.onnx --ml_framework onnx
```

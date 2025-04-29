class ModelRunner:
    def __init__(self, model_path, framework='onnx', device='cpu'):
        self.framework = framework.lower()
        self.device = device.lower()

        if self.framework == 'onnx':
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        elif self.framework == 'pytorch':
            import torch
            self.torch = torch
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
        elif self.framework == 'hailo':
            import hailo_platform
            self.hailo = hailo_platform
            self.model = self.hailo.load_model(model_path)
        else:
            raise ValueError(f"Framework {framework} no soportado")

    def predict(self, input_tensor):
        if self.framework == 'onnx':
            return self.session.run(None, {self.input_name: input_tensor})[0]
        elif self.framework == 'pytorch':
            with self.torch.no_grad():
                tensor = self.torch.tensor(input_tensor).to(self.device)
                output = self.model(tensor)
                return output.cpu().numpy()
        elif self.framework == 'hailo':
            return self.model.infer(input_tensor)

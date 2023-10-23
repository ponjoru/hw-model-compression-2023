import os
import cv2
import numpy as np
from typing import Union, Tuple


class OnnxEngine:
    """
    OnnxEngine
    """
    def __init__(
        self,
        model_path: str,
        num_threads: Union[int, None] = None,
        device: str = 'cpu',
        mean: Tuple[float, float, float] = (0.446139603853, 0.409515678883, 0.395083993673),
        std: Tuple[float, float, float] = (0.288205742836, 0.278144598007, 0.283502370119),
    ):
        """
        :param model_path: path to the .onnx model
        :param num_threads: number of cpu cores to run inference on. Only used if device is set to 'cpu'
        :param device: device to run onnx inference on (cpu or cuda)
        :param mean: img preprocess mean value
        :param std: img preprocess std value
        """
        self.model_path = model_path
        self.mean = mean
        self.std = std

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']

        # num threads in env should be set before onnxruntime import
        if device == 'cpu' and num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        self.session = onnxruntime.InferenceSession(self.model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        if num_threads is not None:
            print(f'Device: {device}, NumThreads: {num_threads}')
        else:
            print(f'Device: {device}')

    def forward(self, x: np.ndarray):
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(x, 1.0, input_size, (0., 0., 0.), swapRB=False)
        out = self.session.run(self.output_names, {self.input_name: blob})
        return out

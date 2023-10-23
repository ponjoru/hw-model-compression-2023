import torch
from src.export import *
from src.upd_scrfd.scrfd import SCRFD_500M
from src.utils import compute_pytorch_latency, compute_numpy_latency


if __name__ == '__main__':
    ts_output_file = '../weights/SCRFD_500M_KPS.pts'
    onnx_output_file = '../weights/SCRFD_500M_KPS.onnx'
    openvino_output_file = '../weights/SCRFD_500M_KPS_openvino.xml'

    input_shape = (640, 640)

    model = SCRFD_500M(nc=1)
    model.load_from_checkpoint('../weights/upd_SCRFD_500M_KPS.pth')
    print('SCRFD_500M_KPS model')
    # # exporting models to TS, ONNX and OpenVino
    # export_torchscript(model, ts_output_file, optimize_for_mobile=False)
    # export_onnx(model, input_shape, output_file=onnx_output_file, dynamic=True, simplify=True)
    # export_openvino(onnx_output_file, openvino_output_file, 'SCRFD_500M')

    # evaluating latency
    print(' PyTorch '.center(100, '='))
    torch_latency = compute_pytorch_latency(model, (1, 3, *input_shape))
    print(f'PyTorch latency: {torch_latency:.2f}ms')

    print(' TorchScript '.center(100, '='))
    model = torch.jit.load(ts_output_file)
    ts_latency = compute_pytorch_latency(model, (1, 3, *input_shape))
    print(f'TorchScript latency: {ts_latency:.2f}ms')

    print(' ONNX '.center(100, '='))
    engine = OnnxEngine(onnx_output_file, num_threads=None, device='cpu')
    onnx_latency = compute_numpy_latency(engine, (*input_shape, 3))
    print(f'ONNX latency: {onnx_latency:.2f}ms')
    engine = OnnxEngine(onnx_output_file, num_threads=1, device='cpu')
    onnx_latency = compute_numpy_latency(engine, (*input_shape, 3))
    print(f'ONNX (1 core) latency: {onnx_latency:.2f}ms')

    print(' OpenVino '.center(100, '='))
    engine = OpenVinoEngine(openvino_output_file)
    ov_latency = compute_numpy_latency(engine, (1, 3, *input_shape))
    print(f'OpenVino latency: {ov_latency:.2f}ms')

    """
    SCRFD_500M_KPS model
    ============================================= PyTorch ==============================================
    PyTorch latency: 48.01ms
    =========================================== TorchScript ============================================
    TorchScript latency: 49.83ms
    =============================================== ONNX ===============================================
    Device: cpu
    ONNX latency: 16.11ms
    Device: cpu, NumThreads: 1
    ONNX (1 core) latency: 35.24ms
    ============================================= OpenVino =============================================
    OpenVino latency: 9.91ms
    """
import torch
import torch.utils.mobile_optimizer as mobile_optimizer


def export_torchscript(
    model,
    output_file='tmp.pts',
    optimize_for_mobile=False,
    optimize_backend='CPU'
):
    ts_model = torch.jit.script(model)
    if optimize_for_mobile:
        ts_model = mobile_optimizer.optimize_for_mobile(ts_model, backend=optimize_backend)
    torch.jit.save(ts_model, output_file)
    print(f'Successfully exported TorchScript model: {output_file}')

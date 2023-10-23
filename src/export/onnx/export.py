import os
import onnx
import torch


def export_onnx(
    model,
    input_shape,
    opset_version=11,
    output_file='tmp.onnx',
    simplify=True,
    dynamic=True,
    verbose=True
):
    if len(input_shape) == 1:
        input_shape = (1, 3, input_shape[0], input_shape[0])
    elif len(input_shape) == 2:
        input_shape = (1, 3) + tuple(input_shape)
    else:
        raise ValueError('invalid input shape')
    tensor_data = torch.rand(input_shape)

    model.eval()

    # Define input and outputs names, which are required to properly define
    # dynamic axes
    input_names = ['input.1']
    output_names = [
        'score_8', 'score_16', 'score_32',
        'bbox_8', 'bbox_16', 'bbox_32',
        'kps_8', 'kps_16', 'kps_32'
    ]
    if simplify or dynamic:
        ori_output_file = output_file.split('.')[0] + "_ori.onnx"
    else:
        ori_output_file = output_file

    # Define dynamic axes for export
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
        dynamic_axes[input_names[0]] = {
            0: '?',
            2: '?',
            3: '?'
        }

    torch.onnx.export(
        model,
        tensor_data,
        ori_output_file,
        keep_initializers_as_inputs=False,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version
    )

    if simplify:
        model = onnx.load(ori_output_file)
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model, output_file)
        os.remove(ori_output_file)

    print(f'Successfully exported ONNX model: {output_file}')

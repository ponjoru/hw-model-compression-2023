import openvino
import openvino.runtime as ov  # noqa
from openvino.tools import mo  # noqa


def export_openvino(onnx_path, output_file, pretty_name):

    def serialize(ov_model, file):
        """Set RT info, serialize and save metadata YAML."""
        ov_model.set_rt_info('SCRFD_500m', ['model_info', 'model_type'])
        ov.serialize(ov_model, file)  # save

    ov_model = mo.convert_model(onnx_path, model_name=pretty_name, framework='onnx', compress_to_fp16=False)  # export

    # if self.args.int8:
    #     assert self.args.data, "INT8 export requires a data argument for calibration, i.e. 'data=coco8.yaml'"
    #     check_requirements('nncf>=2.5.0')
    #     import nncf
    #
    #     def transform_fn(data_item):
    #         """Quantization transform function."""
    #         im = data_item['img'].numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
    #         return np.expand_dims(im, 0) if im.ndim == 3 else im
    #
    #     # Generate calibration data for integer quantization
    #     LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
    #     data = check_det_dataset(self.args.data)
    #     dataset = YOLODataset(data['val'], data=data, imgsz=self.imgsz[0], augment=False)
    #     quantization_dataset = nncf.Dataset(dataset, transform_fn)
    #     ignored_scope = nncf.IgnoredScope(types=['Multiply', 'Subtract', 'Sigmoid'])  # ignore operation
    #     quantized_ov_model = nncf.quantize(ov_model,
    #                                        quantization_dataset,
    #                                        preset=nncf.QuantizationPreset.MIXED,
    #                                        ignored_scope=ignored_scope)
    #     serialize(quantized_ov_model, fq_ov)
    #     return fq, None

    serialize(ov_model, output_file)

    print(f'Successfully exported OpenVINO model: {output_file}')

import torch

from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset, load_metric
from optimum.pipelines import pipeline as ort_pipeline
from optimum.onnxruntime import ORTOptimizer, ORTModelForImageClassification, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig, AutoCalibrationConfig

from src.utils import compute_pipeline_latency


def compute_metrics(predictions, labels):
    accuracy_score = load_metric("accuracy")
    acc = accuracy_score.compute(predictions=predictions, references=labels)
    return acc


def eval_model(pipeline, dataset, limit_n_batches=None):
    class_names = dataset.features['label'].names
    nc = len(class_names)

    label2id = dict(zip(class_names, range(nc)))

    model_preds = []
    gt_labels = []

    for i, item in enumerate(tqdm(dataset)):
        out = pipeline(item['img'])
        cls_id = label2id[out[0]['label']]
        model_preds.append(cls_id)
        gt_labels.append(item['label'])

        if limit_n_batches and i > limit_n_batches:
            break

    acc = compute_metrics(model_preds, gt_labels)
    print(f'Accuracy: {acc["accuracy"]:.4f}')


if __name__ == '__main__':
    num_batches = 50
    vit_cifar = 'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'
    model_name = vit_cifar.split('/')[1]
    save_onnx_path = f'../weights/{model_name}'
    save_opt_onnx_path = f'../weights/optimum_{model_name}'
    save_q_onnx_path = f'../weights/q_{model_name}'
    print(f'{model_name} model')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset('cifar10', split='test')

    # ------------------------- PyTorch -------------------------
    print(' Eval pytorch '.center(100, '='))
    pt_pipe = pipeline(model=vit_cifar, device=device)
    eval_model(pt_pipe, dataset=ds, limit_n_batches=num_batches)
    latency = compute_pipeline_latency(pt_pipe, sample=ds[0]['img'])
    print(f'PyTorch latency: {latency:.4f}ms')

    # ----------------------- Optimum ONNX -------------------------
    print(' Eval Optimum ONNX '.center(100, '='))
    ort_model = ORTModelForImageClassification.from_pretrained(vit_cifar, export=True)
    ort_model.save_pretrained(save_onnx_path)
    ort_pipe_1 = ort_pipeline('image-classification', model=save_onnx_path)
    eval_model(ort_pipe_1, dataset=ds, limit_n_batches=num_batches)
    latency = compute_pipeline_latency(ort_pipe_1, sample=ds[0]['img'])
    print(f'Optimum ONNX latency: {latency:.4f}ms')

    # -------------------- Optimized Optimum ONNX -------------------------
    print(' Eval Optimized Optimum ONNX '.center(100, '='))
    model = ORTModelForImageClassification.from_pretrained(save_onnx_path)
    optimization_config = OptimizationConfig(
        # optimize_for_gpu=True,
        optimization_level=99
    )
    optimizer = ORTOptimizer.from_pretrained(model)
    optimizer.optimize(save_dir=save_opt_onnx_path, optimization_config=optimization_config)
    ort_pipe_2 = ort_pipeline('image-classification', model=save_opt_onnx_path)
    eval_model(ort_pipe_2, dataset=ds, limit_n_batches=num_batches)
    latency = compute_pipeline_latency(ort_pipe_2, sample=ds[0]['img'])
    print(f'Optimum optimized ONNX latency: {latency:.4f}ms')

    # quantization onnx
    # onnx_model = ORTModelForImageClassification.from_pretrained(save_opt_onnx_path)
    # quantizer = ORTQuantizer.from_pretrained(onnx_model)
    # qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    #
    # calibration_dataset = quantizer.get_calibration_dataset(
    #     "cifar10",
    #     dataset_config_name="plain_text",
    #     num_samples=50,
    #     dataset_split="train",
    # )
    # calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    #
    # ranges = quantizer.fit(
    #     dataset=calibration_dataset,
    #     calibration_config=calibration_config,
    #     operators_to_quantize=qconfig.operators_to_quantize,
    # )
    #
    # quantizer.quantize(save_dir=save_q_onnx_path, calibration_tensors_range=ranges, quantization_config=qconfig)
    #
    # ort_pipe_3 = ort_pipeline('image-classification', model=save_q_onnx_path)
    # eval_model(ort_pipe_3, dataset=ds, limit_n_batches=50)
    # latency = compute_pipeline_latency(ort_pipe_3, sample=ds[0]['img'])
    # print(f'Optimum optimized ONNX latency: {latency:.4f}ms')

    """
    vit-base-patch16-224-in21k-finetuned-cifar10 model
    =========================================== Eval pytorch ===========================================
      1%|          | 51/10000 [00:07<23:31,  7.05it/s]
    Accuracy: 0.9615
    PyTorch latency: 130.2932ms
    ======================================== Eval Optimum ONNX =========================================
      1%|          | 51/10000 [00:08<26:09,  6.34it/s]
    Accuracy: 0.9615
    Optimum ONNX latency: 151.8879ms
    =================================== Eval Optimized Optimum ONNX ====================================
      1%|          | 51/10000 [00:07<25:21,  6.54it/s]
    Accuracy: 0.9615
    Optimum optimized ONNX latency: 146.3599ms
    """
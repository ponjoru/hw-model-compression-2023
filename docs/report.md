# Домашние задания по курсу model-compression 2023
Курс: [ссылка](https://github.com/aitalents/model-compression-2023.git) \
Студент: Попов Игорь \
PC: MacBookPro16 2019, 2,3 GHz 8-Core Intel Core i9, 16Gb DDR4, MacOS 14.0

## HW0: Before starting
Была выбрана в качестве основной модель SCRFD_500M_KPS для детекции лиц и 5 ключевых точек. Реализация модели была 
переписана на нативный pytorch с mmdetection2. Текущая реализация скриптуется в TorchScript. 
Веса сконвертированы из оригинальных

NOTE: есть небольшая разница в метриках, так как используется немного другой препроцессинг и evaluator

## HW1: Evaluation
Скрипт: `homeworks/hw1_validation.py` \
Задача: подсчет метрик для выбранной модели, оценить скорость работы на своем железе \
Модель детекции лиц с ключевыми точками: SCRFD_500M (pytorch reimplementation), ([оригинал](https://github.com/deepinsight/insightface/tree/master/detection/scrfd))

| Model      | Dataset  | Latency | easy_AP | medium_AP| hard_AP | CPU                          | 
|------------|----------|---------|---------|----------|---------|------------------------------|
| SCRFD_500M | WIDER_val| 50.68ms | 0.9071  | 0.8805   | 0.6768  | 2,3 GHz 8-Core Intel Core i9 |


## HW2: Quantization and Pruning
Скрипт: `homeworks/hw2_quantization_and_pruning.py` \
Задача: применить методы оптимизации квантизации и прунинга к выбранной модели. Оценить скорость и метрики оптимизированных моделей \
Модель детекции лиц с ключевыми точками: SCRFD_500M_KPS (pytorch reimplementation), ([оригинал](https://github.com/deepinsight/insightface/tree/master/detection/scrfd))

| Model             | Dataset  | Latency | easy_AP | medium_AP| hard_AP | CPU                          | Size  |
|-------------------|----------|---------|---------|----------|---------|------------------------------|-------|
| SCRFD_500M (fp32) | WIDER_val| 50.68ms | 0.9071  | 0.8805   | 0.6768  | 2,3 GHz 8-Core Intel Core i9 | 2.7Mb |
| SCRFD_500M (int8) | WIDER_val| 20.50ms | 0.9023  | 0.8731   | 0.6584  | 2,3 GHz 8-Core Intel Core i9 | 727KB |

Была применена классическая статическая квантизация в int8 с помощью pytorch.quantization, калибрация на train датасете 
целиком с потерей в точности: -1% и быстрее в x2.5 раза на cpu, х4 меньше памяти.

TODO: 
* добавить QAT, чтобы уменьшить ошибку
* применить pruning

## HW3: Weights Clustering
Скрипт: `homeworks/hw3_weights_clusterization.py` \
Задача: применить методы кластеризации весов

Кластеризация весов была реализована для модели MobileNetV2 в фреймворке keras

| Model                    | Latency  | Model Size |
|--------------------------|----------|------------|
| MobileNetV2 (original)   | 138.93ms | 13.19Mb    |
| MobileNetV2 (clustered32)| 151.98ms | 2.90Mb     |

Уменьшение размера весов в x4.5 раза

## HW4: Knowledge distillation
Скрипт: `homeworks/hw4_knowledge_distillation.ipynb`  

Реализован скрипт обучения с дистиляцией: \
Teacher: SCRFD_10G_KPS \
Student: SCRFD_500M_KPS 

Метод дистиляции (a.k.a ReviewKD): [Distilling Knowledge via Knowledge Review](https://arxiv.org/pdf/2104.09044.pdf)

Своего железа (MacBookPro2019) не хватает для обучения, обучение в GoogleColab.

Обучено только 4 эпохи, в статье авторы рекомендуют обучать 100+ эпох

## HW5: Auto-compression (huggingface-optimum)
Скрипт: `homeworks/hw6_export_engines.py` \
Задача: применить метод автоматической оптимизации моделей из пакета [huggingface-optimum](https://huggingface.co/docs/optimum/index)

Для дз выбрана модель из huggingface-transformers `aaraki/vit-base-patch16-224-in21k-finetuned-cifar10`, дообученная на cifar10

| Model                           | Dataset | Latency  | Acc (50 samples) | CPU                          |
|---------------------------------|---------|----------|------------------|------------------------------|
| ViTp16 (PyTorch)                | CIFAR10 | 130.29ms | 0.9615           | 2,3 GHz 8-Core Intel Core i9 |
| ViTp16 (ONNX Optimum)           | CIFAR10 | 151.88ms | 0.9615           | 2,3 GHz 8-Core Intel Core i9 |
| ViTp16 (ONNX Optimum Optimized) | CIFAR10 | 146.35ms | 0.9615           | 2,3 GHz 8-Core Intel Core i9 |

## HW6: Export engines
Скрипт: `homeworks/hw5_autocompression.py` \
Задача: сконвертировать модель в различные инференс движки, замерить скорость работы в них

| Model      | Engine            | Latency | CPU                          |
|------------|-------------------|---------|------------------------------|
| SCRFD_500M | PyTorch           | 48.01ms | 2,3 GHz 8-Core Intel Core i9 |
| SCRFD_500M | TorchScript       | 49.83ms | 2,3 GHz 8-Core Intel Core i9 |
| SCRFD_500M | ONNX              | 16.11ms | 2,3 GHz 8-Core Intel Core i9 |
| SCRFD_500M | ONNX (1 cpu core) | 35.24ms | 2,3 GHz 8-Core Intel Core i9 |
| SCRFD_500M | OpenVINO          | 9.91ms  | 2,3 GHz 8-Core Intel Core i9 |

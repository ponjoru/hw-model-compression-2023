import os
import time
import zipfile
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2


def compute_keras_latency(model, sample):
    times_arr = []
    for i in range(100):
        t0 = time.time()
        _ = model(sample)
        t1 = time.time()

        if i > 10:
            times_arr.append(t1-t0)
    return sum(times_arr) / len(times_arr) * 1000


def apply_clustering(layer):
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {'number_of_clusters': 32, 'cluster_centroids_init': CentroidInitialization.LINEAR}
    if isinstance(layer, (tf.keras.layers.Identity, tf.keras.layers.InputLayer)):
        return layer
    if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.clustering.keras.cluster_weights(layer, **clustering_params)
    return layer


def save_model_file(model):
    _, keras_file = tempfile.mkstemp('.h5')
    model.save(keras_file, include_optimizer=False)
    return keras_file


def get_gzipped_model_size(model):
    # It returns the size of the gzipped model in bytes.
    keras_file = save_model_file(model)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)
    size = os.path.getsize(zipped_file)
    return size * 1e-6


if __name__ == '__main__':
    x = np.random.rand(224, 224, 3)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # ------------------------- Original -------------------------
    print(' Original model '.center(100, '='))
    model = MobileNetV2(weights='imagenet')
    size = get_gzipped_model_size(model)
    print(f'Size: {size:.2f}Mb')
    latency = compute_keras_latency(model, x)
    print(f'Latency: {latency:.2f}ms')

    # ------------------------- Clustering -------------------------
    print(' Clustered model '.center(100, '='))
    clustered_model = tf.keras.models.clone_model(model, clone_function=apply_clustering)
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    size = get_gzipped_model_size(final_model)
    print(f'Size: {size:.2f}Mb')
    latency = compute_keras_latency(final_model, x)
    print(f'Latency: {latency:.2f}ms')

    """
    ========================================== Original model ==========================================
    Size: 13.19Mb
    Latency: 138.93ms
    ========================================= Clustered model ==========================================
    Size: 2.90Mb
    Latency: 151.98ms
    """
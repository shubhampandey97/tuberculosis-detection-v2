"""Transfer-learning models that take raw 0-255 RGB input.

Each model contains its own preprocessing, so a saved model needs no outside scaling:
  vgg, resnet   -> CaffePreprocess (RGB->BGR, subtract ImageNet mean), same as Keras preprocess_input
  efficientnet  -> nothing (Keras EfficientNetB0 rescales and normalises internally)
The base network is frozen; only the classification head is trained.
"""
import keras
import tensorflow as tf
from keras import layers

IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)

BASES = {
    "vgg": keras.applications.VGG16,
    "resnet": keras.applications.ResNet50,
    "efficientnet": keras.applications.EfficientNetB0,
}
NEEDS_CAFFE = {"vgg", "resnet"}
# Last convolutional layer per model (used by Grad-CAM in goal 5).
LAST_CONV_LAYER = {"vgg": "block5_conv3", "resnet": "conv5_block3_out", "efficientnet": "top_conv"}


@keras.saving.register_keras_serializable(package="tb")
class CaffePreprocess(layers.Layer):
    """'caffe' mode of keras.applications.{vgg16,resnet50}.preprocess_input, as a layer."""

    def call(self, x):
        x = tf.reverse(x, axis=[-1])  # RGB -> BGR
        return x - tf.constant(IMAGENET_BGR_MEAN, dtype=x.dtype)


def crop_pixels(size, fraction):
    """Pixels removed from each edge by CentralCrop (same rounding everywhere)."""
    return int(round(fraction * size))


@keras.saving.register_keras_serializable(package="tb")
class CentralCrop(layers.Layer):
    """Goal 10: remove `fraction` of each edge (markers, labels, borders), then resize back to the input size."""

    def __init__(self, fraction=0.12, **kwargs):
        super().__init__(**kwargs)
        self.fraction = float(fraction)

    def call(self, x):
        h, w = x.shape[1], x.shape[2]
        ph, pw = crop_pixels(h, self.fraction), crop_pixels(w, self.fraction)
        return tf.image.resize(x[:, ph:h - ph, pw:w - pw, :], (h, w))  # bilinear

    def get_config(self):
        return {**super().get_config(), "fraction": self.fraction}


@keras.saving.register_keras_serializable(package="tb")
class PerImageStandardize(layers.Layer):
    """Goal 10: per-image z-score rescaled to mean 128 / sd 50 and clipped to 0-255 (backbone's expected range).
    Removes global brightness and contrast, which the data audit found to be source cues."""

    def __init__(self, target_mean=128.0, target_sd=50.0, **kwargs):
        super().__init__(**kwargs)
        self.target_mean, self.target_sd = float(target_mean), float(target_sd)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        sd = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        z = (x - mean) / tf.maximum(sd, 1e-3)  # a constant image maps to target_mean, never NaN
        return tf.clip_by_value(z * self.target_sd + self.target_mean, 0.0, 255.0)

    def get_config(self):
        return {**super().get_config(), "target_mean": self.target_mean, "target_sd": self.target_sd}


def build_model(model_name, img_size=(224, 224), learning_rate=1e-4, weights="imagenet",
                crop_fraction=None, standardize=False):
    """weights="imagenet" for real training; tests pass weights=None to avoid downloading weights.
    crop_fraction / standardize: goal 10 shortcut-removal layers (inside the model, before CaffePreprocess)."""
    if model_name not in BASES:
        raise ValueError(f"Invalid model name '{model_name}', expected one of {list(BASES)}")

    inputs = keras.Input(shape=tuple(img_size) + (3,), name="image_rgb_0_255")
    x = inputs
    if crop_fraction:
        x = CentralCrop(crop_fraction, name="central_crop")(x)
    if standardize:
        x = PerImageStandardize(name="per_image_standardize")(x)
    x = CaffePreprocess(name="caffe_preprocess")(x) if model_name in NEEDS_CAFFE else x

    # Building the base on our tensor keeps all layers in one flat graph,
    # so Grad-CAM can find e.g. "block5_conv3" with model.get_layer().
    base = BASES[model_name](weights=weights, include_top=False, input_tensor=x)
    base.trainable = False  # frozen; BatchNorm layers also run in inference mode

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_tb")(x)

    model = keras.Model(inputs, outputs, name=f"{model_name}_tb")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def set_finetune(model, from_layer, learning_rate):
    """Goal 10 Arm 2: unfreeze every layer from `from_layer` onwards (BatchNorm stays frozen, i.e. in
    inference mode); everything before stays frozen. Recompiles with `learning_rate`."""
    names = [layer.name for layer in model.layers]
    if from_layer not in names:
        raise ValueError(f"Layer {from_layer!r} not in model")
    start = names.index(from_layer)
    for i, layer in enumerate(model.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    compile_model(model, learning_rate)
    return model

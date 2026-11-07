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


def build_model(model_name, img_size=(224, 224), learning_rate=1e-4, weights="imagenet"):
    """weights="imagenet" for real training; tests pass weights=None to avoid downloading weights."""
    if model_name not in BASES:
        raise ValueError(f"Invalid model name '{model_name}', expected one of {list(BASES)}")

    inputs = keras.Input(shape=tuple(img_size) + (3,), name="image_rgb_0_255")
    x = CaffePreprocess(name="caffe_preprocess")(inputs) if model_name in NEEDS_CAFFE else inputs

    # Building the base on our tensor keeps all layers in one flat graph,
    # so Grad-CAM can find e.g. "block5_conv3" with model.get_layer().
    base = BASES[model_name](weights=weights, include_top=False, input_tensor=x)
    base.trainable = False  # frozen; BatchNorm layers also run in inference mode

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_tb")(x)

    model = keras.Model(inputs, outputs, name=f"{model_name}_tb")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model

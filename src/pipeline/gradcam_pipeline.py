import tensorflow as tf
from src.components.gradcam import get_img_array, make_gradcam_heatmap, display_gradcam

MODEL_PATH = "models/tensorflow/vgg_model.h5"
IMAGE_PATH = "data/Dataset of Tuberculosis Chest X-rays Images/TB Chest X-rays/TB.1.jpg"

model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: last conv layer for VGG
# LAST_CONV_LAYER = "block5_conv3"
LAST_CONV_LAYER = "block5_conv2"

img_array = get_img_array(IMAGE_PATH)

heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

display_gradcam(IMAGE_PATH, heatmap)
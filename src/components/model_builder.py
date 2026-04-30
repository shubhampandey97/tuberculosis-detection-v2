import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)

def build_model(model_name):

    if model_name == "resnet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    elif model_name == "vgg":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    elif model_name == "efficientnet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    else:
        raise ValueError("Invalid model name")

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
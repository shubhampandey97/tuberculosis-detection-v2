import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = "data/Dataset of Tuberculosis Chest X-rays Images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    return test_ds

def evaluate_model(model_path):

    print(f"\nEvaluating: {model_path}")

    model = tf.keras.models.load_model(model_path)
    test_ds = load_test_data()

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images)
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())
        y_prob.extend(probs.flatten())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_true, y_prob))


if __name__ == "__main__":
    evaluate_model("models/tensorflow/resnet_model.h5")
    evaluate_model("models/tensorflow/vgg_model.h5")
    evaluate_model("models/tensorflow/efficientnet_model.h5")
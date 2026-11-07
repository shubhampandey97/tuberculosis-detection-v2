from src.components.data_loader import make_dataset
from src.utils.config_loader import load_config

config = load_config()
train_ds = make_dataset(config, "train", training=True)

for images, labels in train_ds.take(1):
    print(images.shape, labels.shape, float(images.numpy().min()), float(images.numpy().max()))

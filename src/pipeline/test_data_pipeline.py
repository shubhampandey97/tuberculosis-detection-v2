from src.components.data_loader import load_datasets, apply_augmentation

DATA_PATH = "data/Dataset of Tuberculosis Chest X-rays Images"

train_ds, val_ds = load_datasets(DATA_PATH)
train_ds = apply_augmentation(train_ds)

for images, labels in train_ds.take(1):
    print(images.shape, labels.shape)
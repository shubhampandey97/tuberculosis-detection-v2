import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.components.torch_dataset import get_dataloaders
from src.components.torch_model import build_model

DATA_PATH = "data/Dataset of Tuberculosis Chest X-rays Images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(model_name):

    train_loader, val_loader = get_dataloaders(DATA_PATH)

    model = build_model(model_name)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"models/torch/{model_name}.pth")

if __name__ == "__main__":
    train("resnet")
    # train("resnet")
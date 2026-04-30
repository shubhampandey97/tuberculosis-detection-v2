import torch.nn as nn
import torchvision.models as models

def build_model(model_name):

    if model_name == "resnet":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif model_name == "vgg":
        model = models.vgg16(weights="IMAGENET1K_V1")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 1)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    else:
        raise ValueError("Invalid model")

    return model
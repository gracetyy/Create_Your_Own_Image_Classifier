import torchvision.models as models
from torch import nn


def create_model(arch, hidden_units):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )
        model.classifier = classifier

    return model

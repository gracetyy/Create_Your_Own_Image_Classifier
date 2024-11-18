import torch
from torchvision import datasets, transforms
import json
import os


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data.class_to_idx


def save_checkpoint(
    model, save_dir, arch, hidden_units, learning_rate, epochs, class_to_idx
):
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "class_to_idx": class_to_idx,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = create_model(checkpoint["arch"], checkpoint["hidden_units"])
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    return model, checkpoint["class_to_idx"]

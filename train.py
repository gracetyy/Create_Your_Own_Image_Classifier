import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from model import create_model
from utils import save_checkpoint, load_data


def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    trainloader, validloader, testloader, class_to_idx = load_data(data_dir)

    model = create_model(arch, hidden_units)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        valid_loss += criterion(logps, labels).item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}"
                )
                running_loss = 0
                model.train()

    save_checkpoint(
        model, save_dir, arch, hidden_units, learning_rate, epochs, class_to_idx
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument("data_directory", type=str, help="Directory of training data")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()
    train(
        args.data_directory,
        args.save_dir,
        args.arch,
        args.learning_rate,
        args.hidden_units,
        args.epochs,
        args.gpu,
    )

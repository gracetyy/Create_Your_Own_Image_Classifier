import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import create_model
from utils import load_checkpoint


def predict(image_path, checkpoint_path, top_k, category_names, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    model, class_to_idx = load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    img = process_image(image_path)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.exp(output)
        top_probs, top_classes = probs.topk(top_k)

    if category_names:
        with open(category_names, "r") as f:
            cat_to_name = json.load(f)
            top_classes = [cat_to_name[str(i)] for i in top_classes[0].cpu().numpy()]

    print("Top K classes:", top_classes)
    print("Probabilities:", top_probs[0].cpu().numpy())


def process_image(image):
    img = Image.open(image)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image.")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K predictions")
    parser.add_argument(
        "--category_names", type=str, help="Path to category to name mapping"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()
    predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

import os
import argparse
import torch
from PIL import Image
from src.dataset.transforms import get_valid_transforms
from src.models.efficientnet import build_efficientnet_b3
from src.models.xception import build_xception

def main():
    parser = argparse.ArgumentParser(description="DeepFake Face Prediction Script")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image file")
    parser.add_argument('--model', type=str, default='efficientnet_b3', choices=['efficientnet_b3', 'xception'],
                        help="Model architecture of checkpoint (default: efficientnet_b3)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model weights checkpoint (.pth)")
    parser.add_argument('--img_size', type=int, default=None, help="Input resolution")
    args = parser.parse_args()

    if args.img_size is None:
        args.img_size = 300 if args.model == 'efficientnet_b3' else 299

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # Initialize model
    print(f"Loading {args.model} architecture...")
    if args.model == 'efficientnet_b3':
        model = build_efficientnet_b3(pretrained=False)
    else:
        model = build_xception(pretrained=False)
        
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Preprocess image
    print("Preprocessing image...")
    img = Image.open(args.image_path).convert('RGB')
    transforms = get_valid_transforms(args.img_size)
    img_tensor = transforms(img).unsqueeze(0).to(device) # Add batch dimension

    # Inference
    print("Executing forward pass...")
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    # Result thresholding
    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else (1.0 - prob)

    print("\n--- Detection Result ---")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == '__main__':
    main()

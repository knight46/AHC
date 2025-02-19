import torch
import argparse
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from net import AIGCClassificationNet


def classify_image(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AIGCClassificationNet(embed_dim=128, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, logits, _ = model(img_tensor, img_tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        print("AI Generated" if pred == 1 else "Real Image")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIGC Image Classifier')
    parser.add_argument('--image', type=str, required=True, default='../datasets/test_data_v2/644a1d24421b4a6cb411fae3d674d75e.jpg')
    parser.add_argument('--weights', type=str, required=True, default='./runs/20250218_195014/best_model.pth')
    args = parser.parse_args()

    classify_image(args.image, args.weights)
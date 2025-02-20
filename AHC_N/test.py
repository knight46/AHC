import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from net import AIGCClassificationNet


def load_model(weights_path, device):
    """
    加载模型权重并返回模型对象
    """
    model = AIGCClassificationNet(embed_dim=128, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, transform):
    """
    加载图片并进行预处理
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def main():
    csv_file = "../datasets/test.csv"
    weights_path = "./runs/20250219_21/best_model.pth"
    output_file = "./runs/20250219_21/test.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(weights_path, device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_paths = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                path = '../datasets/'+row[0].strip()
                image_paths.append(path)

    results = []
    for image_path in image_paths:
        try:
            img_tensor = preprocess_image(image_path, transform).to(device)

            with torch.no_grad():
                _, _, logits, _ = model(img_tensor, img_tensor)
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
            label = pred
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            label = -1
        parts = image_path.split('/')
        path = '/'.join(parts[2:])
        results.append((path, label))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for file_name, label in results:
            writer.writerow([file_name, label])
    print("推理结果已保存到", output_file)


if __name__ == "__main__":
    main()

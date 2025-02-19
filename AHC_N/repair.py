import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from net import AIGCClassificationNet  # 确保net.py在同一目录下


def classify_image(image_path, model_path):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = AIGCClassificationNet(embed_dim=128, num_classes=2)  # 参数需与训练时一致
    model.load_state_dict(torch.load(model_path, map_location=device))  # 确保权重加载到正确设备
    model.to(device)  # 将模型移动到设备
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # 加载并预处理图像
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # 添加batch维度并移动到设备

        # 推理
        with torch.no_grad():
            _, _, logits, _ = model(img_tensor, img_tensor)  # 使用相同图片作为双输入
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return "AI Generated" if pred == 1 else "Real Image"
        # # 输出结果
        # print("AI Generated" if pred == 1 else "Real Image")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    image = '../datasets/test_data_v2/ef16e1cfdf3b46679aea4453853d89bd.jpg'
    weights = './runs/20250218_195014/best_model.pth'
    img = cv2.imread(image)
    res = classify_image(image, weights)
    cv2.imshow(res, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

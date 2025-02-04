import torch
from net import BaseFeatureExtractor, DetailFeatureExtractor, Classifier_Head  # 假设你的代码保存在 net.py 文件中

batch_size = 8  # 批量大小
channels = 64  # 输入通道数（dim）
height = 32  # 输入高度
width = 32  # 输入宽度
num_heads = 8  # 注意力头数

# 创建一个随机输入张量，模拟模型的输入
input_tensor = torch.randn(batch_size, channels, height, width)


def BaseFeatureExtractor_test():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 创建 BaseFeatureExtractor 实例
    base_feature_extractor = BaseFeatureExtractor(
        dim=channels,  # 输入通道数
        num_heads=num_heads,  # 注意力头数
        ffn_expansion_factor=2.0,  # MLP 扩展因子
        qkv_bias=False  # 是否使用 QKV 偏置
    )
    base_feature_extractor.eval()

    # 将输入张量传递给 BaseFeatureExtractor
    with torch.no_grad():  # 禁用梯度计算
        output_tensor = base_feature_extractor(input_tensor)

    # 检查输出的形状和类型
    print("输入张量的形状:", input_tensor.shape)
    print("输出张量的形状:", output_tensor.shape)
    print("输出张量的类型:", output_tensor.dtype)

    # 验证输出是否合理
    assert output_tensor.shape == input_tensor.shape, "输出形状与输入形状不匹配！"
    assert not torch.isnan(output_tensor).any(), "输出张量包含 NaN 值！"
    assert not torch.isinf(output_tensor).any(), "输出张量包含 Inf 值！"

    print("测试通过！BaseFeatureExtractor 可用。")
    return output_tensor

def DetailFeatureExtractor_test():
    detail_feature_extractor = DetailFeatureExtractor(num_layers=3)
    detail_feature_extractor.eval()
    with torch.no_grad():  # 禁用梯度计算
        output_tensor = detail_feature_extractor(input_tensor)
    print("输入张量的形状:", input_tensor.shape)
    print("输出张量的形状:", output_tensor.shape)
    print("输出张量的类型:", output_tensor.dtype)

    # 验证输出是否合理
    assert output_tensor.shape == input_tensor.shape, "输出形状与输入形状不匹配！"
    assert not torch.isnan(output_tensor).any(), "输出张量包含 NaN 值！"
    assert not torch.isinf(output_tensor).any(), "输出张量包含 Inf 值！"

    print("测试通过！DetailFeatureExtractor 可用。")
    return output_tensor

if __name__ == "__main__":
    output_feature = torch.cat((BaseFeatureExtractor_test(),DetailFeatureExtractor_test()), dim=1)
    classhead = Classifier_Head(in_features=128, num_classes=3, dropout=0)
    x = torch.softmax(classhead(output_feature), dim=1)
    print(x.shape)
    print(x)
    print(torch.argmax(x, dim=1))


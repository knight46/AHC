import torch
from net import BaseFeatureExtractor  # 假设你的代码保存在 net.py 文件中

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 定义输入参数
batch_size = 2  # 批量大小
channels = 64   # 输入通道数（dim）
height = 32     # 输入高度
width = 32      # 输入宽度
num_heads = 8   # 注意力头数

# 创建一个随机输入张量，模拟模型的输入
input_tensor = torch.randn(batch_size, channels, height, width)

# 创建 BaseFeatureExtractor 实例
base_feature_extractor = BaseFeatureExtractor(
    dim=channels,  # 输入通道数
    num_heads=num_heads,  # 注意力头数
    ffn_expansion_factor=2.0,  # MLP 扩展因子
    qkv_bias=False  # 是否使用 QKV 偏置
)

# 将模型设置为评估模式（如果需要）
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
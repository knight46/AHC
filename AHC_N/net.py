import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding2D(nn.Module):
    """轻量化的二维位置编码"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.pos_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

    def forward(self, x: Tensor):
        pos_feature = self.pos_conv(x)
        return x + pos_feature


class SharedTransformer(nn.Module):
    """轻量化的共享编码器"""

    def __init__(self, in_channels=3, embed_dim=128, num_heads=4):
        super().__init__()
        # 更高效的patch嵌入
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=4, padding=3),  # 下采样
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        self.pos_encoder = PositionalEncoding2D(embed_dim)

        # 精简Transformer配置
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=2  # 减少层数
        )

    def forward(self, x: Tensor):
        # 输入形状: [B, C, H, W]
        x = self.patch_embed(x)  # [B, E, H/4, W/4]
        x = self.pos_encoder(x)

        # 转换为序列
        B, E, H, W = x.size()
        x = x.view(B, E, -1).permute(0, 2, 1)  # [B, L, E]

        # Transformer处理
        x = self.transformer(x)  # [B, L, E]

        # 自适应池化
        x = x.mean(dim=1)  # [B, E]
        return x


class MLPClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class AIGCClassificationNet(nn.Module):
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.shared_encoder = SharedTransformer(embed_dim=embed_dim)
        self.classifier = MLPClassifier(embed_dim, num_classes)

    def forward(self, real_images: Tensor, aigc_images: Tensor):
        real_emb = self.shared_encoder(real_images)
        aigc_emb = self.shared_encoder(aigc_images)
        real_logits = self.classifier(real_emb)
        aigc_logits = self.classifier(aigc_emb)
        return real_emb, aigc_emb, real_logits, aigc_logits

def supervised_contrastive_loss(real_emb: Tensor, aigc_emb: Tensor, temperature=0.1, eps=1e-8):

    device = real_emb.device
    batch_size = real_emb.size(0)
    real_emb = F.normalize(real_emb, p=2, dim=1)
    aigc_emb = F.normalize(aigc_emb, p=2, dim=1)

    embeddings = torch.cat([real_emb, aigc_emb], dim=0)
    labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)], dim=0).to(device)

    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e4)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_positive = labels_equal.float() - torch.eye(2 * batch_size, device=device)

    exp_sim = torch.exp(sim_matrix)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=eps)

    log_prob = sim_matrix - torch.log(denom)

    num_positives = mask_positive.sum(dim=1)
    loss = - (mask_positive * log_prob).sum(dim=1) / (num_positives + eps)

    return loss.mean()

# def main():
#     batch_size = 8
#     image_size = (256, 256)
#
#     real_images = torch.randn(batch_size, 3, *image_size)
#     aigc_images = torch.randn(batch_size, 3, *image_size)
#
#     model = AIGCClassificationNet(embed_dim=256, num_classes=2)
#
#     real_emb, aigc_emb, _, _ = model(real_images, aigc_images)
#
#     loss = info_nce_loss(real_emb, aigc_emb)
#
#     print(f"对比损失: {loss.item()}")

# if __name__ == "__main__":
#     main()

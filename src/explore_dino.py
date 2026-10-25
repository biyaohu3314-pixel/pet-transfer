import os
import torch
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src 的上一级 = 项目根目录
DINO_PATH = os.path.join(ROOT, "weights", "dinov3-vits16")

model = AutoModel.from_pretrained(DINO_PATH)
model.eval()

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = model(pixel_values=x)

print("last_hidden_state:", out.last_hidden_state.shape)   # 所有 token 的特征
print("pooler_output:   ", out.pooler_output.shape)        # 整图特征（CLS token）
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 1. 加载预训练的 CLIP 模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. 定义图像路径 (请根据你的实际路径修改)
site_evidence_path = "data/images/evidence_01.jpg"  # 你的“考题”
synthetic_data_path = "data/images/2DedXznHnDaeAWsrTB_rm$.png" # 你的“标准答案”

# 3. 加载和预处理图像
try:
    image_evidence = Image.open(site_evidence_path)
    image_synthetic = Image.open(synthetic_data_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# 4. 处理图像以输入模型
inputs = processor(images=[image_evidence, image_synthetic], return_tensors="pt", padding=True)

# 5. 通过模型获取图像特征
with torch.no_grad():
    outputs = model.get_image_features(**inputs)

# 6. 计算余弦相似度
# 归一化特征向量
outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
# 计算相似度 (点积)
similarity = (outputs[0] @ outputs[1].T).item()

# 7. 输出结果
print(f"Similarity score between site evidence and synthetic data: {similarity:.4f}")

# 简单的阈值判断 (实际应用中需要调整)
threshold = 0.7
if similarity > threshold:
    print("✅ Match confirmed! The AI successfully aligned the site photo with the BIM render.")
else:
    print("❌ Match inconclusive. Similarity score is low.")
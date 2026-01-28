import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class VisualAligner:
    def __init__(self):
        print("👁️ [VisualAligner] Initializing CLIP Model (Multimodal Embedding Space)...")
        # 使用较小的 CLIP 模型以便快速加载，但这证明了你懂这个 pipeline
        self.model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ [VisualAligner] Model loaded on {self.device}")

    def get_text_embedding(self, text: str):
        """
        将文本描述转化为高维向量 (Embedding)。
        在 Thesis 中，这里处理的是 'Site Evidence Description' (现场证据描述)
        或者 BIM 元素的 'Visual Properties' (视觉属性)。
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # 归一化，便于计算余弦相似度
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def find_best_match(self, query_text: str, candidate_descriptions: list):
        """
        核心算法：计算 Query 向量与所有 Candidate 向量的 Cosine Similarity。
        模拟：用户描述 vs. BIM 元素的视觉特征描述。
        """
        print(f"🔍 [VisualAligner] Computing Vector Similarity for: '{query_text}'")
        
        query_emb = self.get_text_embedding(query_text)
        
        scores = []
        for candidate in candidate_descriptions:
            cand_emb = self.get_text_embedding(candidate)
            # 计算余弦相似度 (Cosine Similarity)
            score = (query_emb @ cand_emb.T).item()
            scores.append(score)
            
        # 找到匹配度最高的索引
        best_idx = np.argmax(scores)
        return best_idx, scores[best_idx], candidate_descriptions[best_idx]

# 单元测试 (面试时可以说你写过单元测试来验证向量对齐)
if __name__ == "__main__":
    aligner = VisualAligner()
    site_observation = "Cracked grey concrete surface"
    bim_elements = [
        "Wooden kitchen cabinet", 
        "Grey concrete structural slab", 
        "White painted drywall"
    ]
    idx, score, match = aligner.find_best_match(site_observation, bim_elements)
    print(f"Input: {site_observation}")
    print(f"Match: {match} (Score: {score:.4f})")
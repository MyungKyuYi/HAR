import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_dim)

    def forward(self, images):
        return self.model(images)

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용
        return self.fc(cls_embedding)


class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)  # Scaling Factor

    def forward(self, images, input_ids, attention_mask):
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        # L2 정규화
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Similarity 계산 (Cosine Similarity)
        logits = (image_features @ text_features.T) * self.logit_scale.exp()

        return logits

def clip_loss(image_features, text_features, temperature=0.07):
    batch_size = image_features.shape[0]

    # Cosine Similarity 계산
    logits_per_image = (image_features @ text_features.T) / temperature
    logits_per_text = logits_per_image.T
    labels = torch.arange(batch_size, device=image_features.device)

    loss = (F.cross_entropy(logits_per_image, labels) + 
            F.cross_entropy(logits_per_text, labels)) / 2
    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"

#모델 학습 (Training)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 초기화
model = CLIP(embed_dim=512).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, input_ids, attention_mask in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward Pass
        logits = model(images, input_ids, attention_mask)

        # Contrastive Loss 계산
        image_features = model.vision_encoder(images)
        text_features = model.text_encoder(input_ids, attention_mask)
        loss = clip_loss(image_features, text_features)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

####### TEST

# 테스트할 텍스트 프롬프트 (분류할 카테고리)
categories = ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a person"]
text_inputs = clip.tokenize(categories).to(device)  # 텍스트 토큰화

# 테스트할 이미지 로드
image_path = "test_image.jpg"  # 테스트할 이미지 파일
image = Image.open(image_path).convert("RGB")

# CLIP이 요구하는 전처리 수행
image_input = preprocess(image).unsqueeze(0).to(device)

# 이미지와 텍스트 임베딩 벡터 생성
with torch.no_grad():
    image_features = model.encode_image(image_input)  # 이미지 임베딩
    text_features = model.encode_text(text_inputs)  # 텍스트 임베딩

# L2 정규화 (벡터 크기 조정)
image_features /= image_features.norm(dim=1, keepdim=True)
text_features /= text_features.norm(dim=1, keepdim=True)

# 코사인 유사도 계산
similarity = (image_features @ text_features.T)[0]  # (1, num_classes) 크기
prediction = similarity.argmax().item()  # 가장 유사한 카테고리 선택

# 결과 출력
print(f"Predicted class: {categories[prediction]}")

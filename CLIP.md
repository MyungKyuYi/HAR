import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# =========================== 1. 데이터셋 정의 ===========================
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, text_descriptions, transform=None):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.text_descriptions[idx]

        if self.transform:
            image = self.transform(image)
        
        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return image, text_tokens["input_ids"].squeeze(0), text_tokens["attention_mask"].squeeze(0)

# =========================== 2. Vision Encoder 정의 ===========================
class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_dim)

    def forward(self, images):
        return self.model(images)

# =========================== 3. Text Encoder 정의 ===========================
class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용
        return self.fc(cls_embedding)

# =========================== 4. CLIP 모델 정의 ===========================
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

# =========================== 5. Contrastive Loss 정의 ===========================
def clip_loss(image_features, text_features, temperature=0.07):
    batch_size = image_features.shape[0]

    # Cosine Similarity 계산
    logits_per_image = (image_features @ text_features.T) / temperature
    logits_per_text = logits_per_image.T
    labels = torch.arange(batch_size, device=image_features.device)

    loss = (F.cross_entropy(logits_per_image, labels) + 
            F.cross_entropy(logits_per_text, labels)) / 2
    return loss

# =========================== 6. 데이터셋 로드 ===========================
image_paths = ["data/cat.jpg", "data/dog.jpg"]  # 이미지 경로
text_descriptions = ["A photo of a cat", "A photo of a dog"]  # 텍스트 설명

dataset = ImageTextDataset(image_paths, text_descriptions)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# =========================== 7. 모델 학습 ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIP(embed_dim=512).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, input_ids, attention_mask in dataloader:
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

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

# =========================== 8. 평가 (Inference) ===========================
def encode_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])
    image = transform(image).unsqueeze(0).to(device)
    return model.vision_encoder(image)

def encode_text(text, model, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    input_ids, attention_mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
    return model.text_encoder(input_ids, attention_mask)

image_embedding = encode_image("data/cat.jpg", model, device)
text_embedding = encode_text("A photo of a cat", model, device)

# Cosine Similarity 측정
similarity = F.cosine_similarity(image_embedding, text_embedding)
print(f"Image-Text Similarity: {similarity.item():.4f}")

#---------------- TEST 

# CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Zero-shot 분류할 텍스트 프롬프트 정의
categories = ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a person"]
text_inputs = clip.tokenize(categories).to(device)

# 테스트할 이미지 로드 및 전처리
image_path = "test_image.jpg"
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

# 이미지 & 텍스트 임베딩 추출
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# 벡터 정규화
image_features /= image_features.norm(dim=1, keepdim=True)
text_features /= text_features.norm(dim=1, keepdim=True)

# 코사인 유사도 계산
similarity = (image_features @ text_features.T)[0]
prediction = similarity.argmax().item()

# 결과 출력
print(f"Predicted class: {categories[prediction]}")



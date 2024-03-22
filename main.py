# %%
# %%
import os
import cv2
import glob
import timm
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vit_pytorch.vivit import ViT
from tqdm.autonotebook import tqdm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.models as models

# %%
with open('train.txt','r') as f:
    data_labels = f.readlines()

# %%
train_data, valid_data = train_test_split(data_labels, test_size=0.02, random_state=42)

# %%
class CFG:
    debug = False
    video_path = "/media/milkyway/HTreasury/Dataset/AIG/VQA/training"
    captions_path = "."
    batch_size = 16
    num_workers = 4
    head_lr = 1e-3
    video_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    classification_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_size = 224
    video_len = 16
    video_embedding = 400
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

# %%
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

# %%
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, video_data, tokenizer, transforms, video_len):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        video_filenames = []
        mos = []
        prompts = []
        for vid_data in video_data:
            raw_data = vid_data.split('|')
            video_filenames.append(raw_data[0])
            prompts.append(raw_data[1])
            mos.append(float(raw_data[2].replace('\n','')))

        self.video_filenames = video_filenames
        self.mos = mos
        self.captions = list(prompts)
        self.encoded_captions = tokenizer(
            list(prompts), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms
        self.video_len = video_len

    def read_video_as_tensor(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frames = []
        padd_frame = np.zeros((CFG.frame_size, CFG.frame_size, 3), dtype=np.uint8)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = cv2.resize(frame,(CFG.frame_size, CFG.frame_size))
            frame_tensor = torch.from_numpy(frame)  # Convert to PyTorch tensor
            frames.append(frame_tensor)
            frame_idx += 1
        while frame_idx < CFG.video_len:
            frame_tensor = torch.from_numpy(padd_frame)  # Convert to PyTorch tensor
            frames.append(frame_tensor)
            frame_idx += 1

        cap.release()
        video_tensor = torch.stack(frames)  # Stack frames to create video tensor
        return video_tensor

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        video = self.read_video_as_tensor(f"{CFG.video_path}/{self.video_filenames[idx]}")
        
        item['video'] = torch.tensor(video).permute(3, 0, 1, 2).float()
        item['caption'] = self.captions[idx]
        item['mos'] = torch.tensor(self.mos[idx]/100).float()

        return item


    def __len__(self):
        return len(self.captions)

# %%
video_files = glob.glob("/media/milkyway/HTreasury/Dataset/AIG/VQA/val/*mp4")

# %%
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

# %%
class VideoEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = models.video.r3d_18(pretrained=True)
        # video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

# %%
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


# %%
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# %%
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        video_embedding=CFG.video_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.video_encoder = VideoEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=video_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.classification_model = torch.nn.Sequential( 
                torch.nn.Linear(in_features = 256, out_features = 1), 
                torch.nn.Sigmoid() 
            )
        
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        # Getting Image and Text Features
        video_features = self.video_encoder(batch["video"])
        mos_scores = batch["mos"]
        
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        video_embeddings = self.image_projection(video_features)
        text_embeddings = self.text_projection(text_features)
        

        # output_linear = self.classification_model(video_embeddings) 
        
        # Calculating the Loss
        embeddings_similarity = (self.cos(video_embeddings, text_embeddings)+1)/2
        vec_product = video_embeddings*text_embeddings
        magnitude1 = torch.norm(video_embeddings)
        magnitude2 = torch.norm(text_embeddings)
        normalized_dot = vec_product / (magnitude1 * magnitude2)
        output_linear = self.classification_model(normalized_dot) 
        # print(video_embeddings)
        sim_loss = self.mse_loss(embeddings_similarity, mos_scores)
        cls_loss = self.mse_loss(output_linear, mos_scores)
        
        loss = (sim_loss + (3*cls_loss))/4
        
        return loss


# %%
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# %%
def build_loaders(data, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        data,
        tokenizer=tokenizer,
        transforms=transforms,
        video_len=CFG.video_len
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

# %%
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["video"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["video"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


# %%
train_df =train_data
valid_df = valid_data

# %%
def main():
    # train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.video_encoder.parameters(), "lr": CFG.video_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
        # {"params": model.image_projection.parameters(), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
        {"params": model.classification_model.parameters(), "lr": CFG.classification_encoder_lr}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

        if epoch % 10 == 0 :
            print("#############")
            torch.save(model.state_dict(), "model/model.pth")
            print("model saved")
            print("model saved")
            print("#############")

    
    torch.save(model.state_dict(), "model/model.pth")
    print("finale model saved")
    

# %%
# %%
main()



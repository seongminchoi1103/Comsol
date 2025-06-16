import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# Hyper-parameters
num_epoch = 10000
batch_size = 32
lr_d = 0.0001 * 0.5  # D 학습률 절반으로 조절
lr_g = 0.0001
img_channels = 1
img_size = 56
condition_size = 7
noise_size = 100

gpu_id = 1
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f"Now using device: {device}")

dir_name = "/home/gtx2080tix3/seongmin/comsol/data_comsol"
output_name = "/home/gtx2080tix3/seongmin/comsol/result_csvmtx_complete"
os.makedirs(output_name, exist_ok=True)

def evaluate_generator_r2_mae(generator, dataset):
    generator.eval()
    preds, targets = [], []
    for image, cond_vec in dataset:
        cond_vec = cond_vec.unsqueeze(0).to(device)
        noise = torch.randn(1, noise_size).to(device)
        with torch.no_grad():
            gen_img = generator(noise, cond_vec).squeeze(0).cpu().numpy()
        gt_img = image.squeeze(0).cpu().numpy()

        preds.append(gen_img.flatten())
        targets.append(gt_img.flatten())

    preds = np.array(preds)
    targets = np.array(targets)

    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)

    generator.train()
    return r2, mae

def normalize_cond_vec(cond_vec):
    cond_vec[0] = 1.0
    cond_vec[1:4] = cond_vec[1:4] / 3000.0
    cond_vec[4:7] = cond_vec[4:7] / 180.0
    return cond_vec

def denormalize_cond_vec(cond_vec):
    cond_vec = cond_vec.clone()  # 원본 보존용 복사본
    cond_vec[0] = 1.0  # 0번째는 항상 1.0 (그대로)
    cond_vec[1:4] = cond_vec[1:4] * 3000.0
    cond_vec[4:7] = cond_vec[4:7] * 180.0
    return cond_vec
    
import pandas as pd  # 맨 위 import 부분에 추가해주세요

class CustomCSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # pandas로 CSV 읽기 (헤더 없음)
        data = pd.read_csv(file_path, header=None).values
        # 첫 행, 첫 열 제거
        data = data[1:, 1:]
        # float 변환
        data = data.astype(float)
        # NaN -> 0 처리
        data = np.nan_to_num(data, nan=0.0)
        assert data.shape == (img_size, img_size), f"Invalid shape: {data.shape}"

        image = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, 56, 56)

        base_name = os.path.splitext(file_name)[0]
        cond_strs = base_name.split('_')
        assert len(cond_strs) == condition_size, f"Filename condition vector size mismatch: {file_name}"

        cond_vec = torch.tensor([float(c) for c in cond_strs], dtype=torch.float32)
        cond_vec = normalize_cond_vec(cond_vec)

        return image, cond_vec

# 데이터셋 로딩
custom_dataset = CustomCSVImageDataset(dir_name)
data_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# --- Generator with CNN transpose layers ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(noise_size + condition_size, 256 * 7 * 7)
        self.deconv_layers = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 28x28 -> 56x56
        )

    def forward(self, noise, cond_vec):
        x = torch.cat([noise, cond_vec], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.deconv_layers(x)
        return x

# --- Discriminator with CNN ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # condition vector를 공간 차원(56x56)에 맞게 반복 확장해서 이미지와 concat
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_size, img_size * img_size),
            nn.LeakyReLU(0.2),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 4, 2, 1),  # 56x56 -> 28x28
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),               # 28x28 -> 14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),              # 14x14 -> 7x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond_vec):
        # cond_vec: (B, condition_size) -> (B, 1, 56, 56)
        cond_map = self.condition_embed(cond_vec)
        cond_map = cond_map.view(-1, 1, img_size, img_size)
        x = torch.cat([img, cond_map], dim=1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

def check_condition(_generator):
    _generator.eval()
    cond_vectors = [
        [60, 2700, 300, 0, 0, 180, 0],
        [60, 1500, 600, 0, 0, 90, 0],
        [60, 3000, 450, 0, 0, 45, 0],
    ]
    cond_tensors = []
    for cond_vec_list in cond_vectors:
        cond_vec = torch.tensor(cond_vec_list, dtype=torch.float32)
        cond_vec = normalize_cond_vec(cond_vec).unsqueeze(0).to(device)
        cond_tensors.append(cond_vec)
    cond_batch = torch.cat(cond_tensors, dim=0)
    noise = torch.randn(len(cond_vectors), noise_size).to(device)
    with torch.no_grad():
        fake_images = _generator(noise, cond_batch)
    save_image(fake_images, os.path.join(output_name, 'CGAN_test_result.png'), normalize=True)
    _generator.train()

# --- 학습 루프 ---
def train():
    best_r2 = -float('inf')
    best_mae = float('inf')
    log_file = os.path.join(output_name, "best_model_log.txt")
    all_log_file = os.path.join(output_name, "full_log.txt")

    for epoch in range(num_epoch):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epoch}]")
        for images, cond_vec in loop:
            batch_size = images.size(0)
            real_label = torch.full((batch_size, 1), 0.9).to(device)  # label smoothing
            fake_label = torch.zeros(batch_size, 1).to(device)
            images = images.to(device)
            cond_vec = cond_vec.to(device)

            # --- Train Discriminator (1 step) ---
            d_optimizer.zero_grad()
            real_pred = discriminator(images, cond_vec)
            d_real_loss = criterion(real_pred, real_label)

            noise = torch.randn(batch_size, noise_size).to(device)
            fake_images = generator(noise, cond_vec)
            fake_pred = discriminator(fake_images.detach(), cond_vec)
            d_fake_loss = criterion(fake_pred, fake_label)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            d_optimizer.step()

            # --- Train Generator (2 steps) ---
            g_total_loss = 0
            for _ in range(2):
                g_optimizer.zero_grad()
                noise = torch.randn(batch_size, noise_size).to(device)
                fake_images = generator(noise, cond_vec)
                fake_pred = discriminator(fake_images, cond_vec)
                g_loss = criterion(fake_pred, real_label)
                g_loss.backward()
                g_optimizer.step()
                g_total_loss += g_loss.item()

            avg_g_loss = g_total_loss / 2
            loop.set_postfix(d_loss=d_loss.item(), g_loss=avg_g_loss)

        # --- Evaluate ---
        r2, mae = evaluate_generator_r2_mae(generator, custom_dataset)

        # 전체 로그 기록
        with open(all_log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: R2={r2:.4f}, MAE={mae:.4f}\n")

        # Best model 저장 조건
        if r2 > best_r2 and mae < best_mae:
            best_r2 = r2
            best_mae = mae
            best_path = os.path.join(output_name, 'best_model.pt')
            torch.save(generator.state_dict(), best_path)
            with open(log_file, 'a') as f:
                f.write(f"Best Updated at Epoch {epoch+1}: R2={r2:.4f}, MAE={mae:.4f}\n")

if __name__ == "__main__":
    train()
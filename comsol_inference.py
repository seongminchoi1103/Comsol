import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# === 1. 정규화 함수 정의 ===
def normalize_minmax(data, data_min=None, data_max=None):
    if data_min is None:
        data_min = data.min()
    if data_max is None:
        data_max = data.max()
    norm = (data - data_min) / (data_max - data_min + 1e-8)
    return norm, data_min, data_max

def denormalize_minmax(norm_data, data_min, data_max):
    return norm_data * (data_max - data_min + 1e-8) + data_min

# === CSV → PNG 변환 함수 ===
def csv_to_png(csv_path, png_path):
    # pandas로 CSV 읽기 (헤더가 있을 경우 기본적으로 첫 행을 헤더로 인식)
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)  # DataFrame → numpy array, float 변환

    plt.imshow(data, cmap='viridis')
    plt.colorbar()
    plt.savefig(png_path)
    plt.close()

# === PNG → numpy array 로딩 (GAN input용) ===
def load_png_as_tensor(png_path):
    img = Image.open(png_path).convert('L')  # 흑백 모드
    img_np = np.array(img).astype(np.float32)
    # 0~255 → 0~1 정규화
    img_np_norm = img_np / 255.0
    # 텐서 변환, (1, 56, 56)
    img_tensor = torch.tensor(img_np_norm).unsqueeze(0)
    return img_tensor

# === 데이터 처리: 폴더 내 CSV → PNG 변환 후 PNG 로드 ===
def process_all_files_as_png(folder_path, temp_png_folder):
    condition_vectors = []
    input_tensors = []

    # PNG 임시 저장 폴더 없으면 생성
    if not os.path.exists(temp_png_folder):
        os.makedirs(temp_png_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 조건 벡터 추출 (예: '60_279_2700_0_0_25_0.csv')
            condition_vector = [int(val) for val in filename.replace('.csv', '').split('_')]
            condition_vectors.append(condition_vector)

            # CSV → PNG 변환
            png_path = os.path.join(temp_png_folder, filename.replace('.csv', '.png'))
            csv_to_png(file_path, png_path)

            # PNG → Tensor 로드
            img_tensor = load_png_as_tensor(png_path)
            input_tensors.append(img_tensor)

    input_tensors = torch.stack(input_tensors)  # (N, 1, 56, 56)
    condition_vectors = np.array(condition_vectors)

    return condition_vectors, input_tensors

# === 설정 ===
latent_dim = 100
condition_dim = 7
image_size = 56
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(Generator, self).__init__()
        input_dim = latent_dim + condition_dim
        self.fc = nn.Linear(input_dim, 128 * 14 * 14)
        self.gen = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # (B, 128, 28, 28)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # (B, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # (B, 64, 56, 56)
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # (B, 1, 56, 56)
            nn.Tanh()
        )

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x).view(-1, 128, 14, 14)
        img = self.gen(x)
        return img


class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()
        self.condition_fc = nn.Linear(condition_dim, 56 * 56)
        self.dis = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),  # (B, 2, 56, 56)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 1)
        )

    def forward(self, img, condition):
        cond_map = self.condition_fc(condition).view(-1, 1, 56, 56)
        x = torch.cat([img, cond_map], dim=1)  # (B, 2, 56, 56)
        return self.dis(x)

# === 2. 데이터 로딩 및 정규화 적용 ===
folder_path = '/mnt/c/Users/user/CGAN-PyTorch/comsol/data_comsol'
temp_png_folder = '/mnt/c/Users/user/CGAN-PyTorch/comsol/temp_png'

condition_vectors, input_tensors = process_all_files_as_png(folder_path, temp_png_folder)

# input_tensors는 (N, 1, 56, 56), 0~1로 정규화되어 있음
input_tensors = input_tensors.to(device)
condition_vectors = torch.tensor(condition_vectors, dtype=torch.float32).to(device)

# 하이퍼파라미터 설정
batch_size = 32
num_epoch = 200
latent_dim = 100
lr = 2e-4

# 모델 초기화
generator = Generator(latent_dim, condition_dim).to(device)
discriminator = Discriminator(condition_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# === 학습 루프 ===
for epoch in range(num_epoch):
    perm = torch.randperm(len(condition_vectors))
    for i in range(0, len(condition_vectors), batch_size):
        idx = perm[i:i+batch_size]
        real_imgs = input_tensors[idx]
        conds = condition_vectors[idx]

        valid = torch.ones((real_imgs.size(0), 1), device=device)
        fake = torch.zeros((real_imgs.size(0), 1), device=device)

        # === Generator step ===
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z, conds)
        pred_fake = discriminator(gen_imgs, conds)
        loss_G = criterion(pred_fake, valid)
        loss_G.backward()
        optimizer_G.step()

        # === Discriminator step ===
        optimizer_D.zero_grad()
        pred_real = discriminator(real_imgs, conds)
        pred_fake = discriminator(gen_imgs.detach(), conds)
        loss_D_real = criterion(pred_real, valid)
        loss_D_fake = criterion(pred_fake, fake)
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{num_epoch}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

# === 인퍼런스 ===
def generate_matrix_with_condition(generator, condition, latent_dim, device):
    generator.eval()
    z = torch.randn(1, latent_dim).to(device)
    condition_tensor = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        gen_img = generator(z, condition_tensor)
    gen_img_np = gen_img.squeeze().cpu().numpy()
    gen_img_np = (gen_img_np + 1) / 2  # Tanh 역정규화: [-1,1] → [0,1]
    return gen_img_np

def save_generated_matrix_to_csv(matrix, output_path):
    # (56, 56) 형태 예상
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(output_path, matrix, delimiter=',')
    print(f"매트릭스가 {output_path}에 저장되었습니다.")

# === 예시 조건 벡터 ===
condition_input = [60, 279, 2700, 0, 0, 25, 0]  # 예시 조건 (파일 이름에서 추출된 벡터)

# === 생성 및 저장 ===
output_file_path = '/mnt/c/Users/user/CGAN-PyTorch/result_comsol/generated_matrix.csv'
generated_matrix = generate_matrix_with_condition(generator, condition_input, latent_dim, device)
save_generated_matrix_to_csv(generated_matrix, output_file_path)

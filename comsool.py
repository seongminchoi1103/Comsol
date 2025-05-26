import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

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

# === 데이터 처리 ===
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    df_matrix = df.drop(0, axis=0).drop(0, axis=1).values  # 첫 번째 행과 첫 번째 열 삭제
    df_matrix = np.nan_to_num(df_matrix, nan=0.0)  # NaN을 0으로 대체
    return df_matrix

def process_all_files(folder_path):
    condition_vectors = []
    input_matrices = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            condition_vector = [int(val) for val in filename.replace('.csv', '').split('_')]
            condition_vectors.append(condition_vector)
            df_matrix = load_and_process_csv(file_path)
            input_matrices.append(df_matrix)

    return condition_vectors, input_matrices

# === 설정 ===
latent_dim = 100
condition_dim = 7
image_size = 56
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Generator ===
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, image_size * image_size),
        )

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x)
        return x.view(x.size(0), 1, image_size, image_size)

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(condition_dim + image_size * image_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, condition, input_img):
        if len(input_img.shape) == 4:  # (B, 1, 56, 56)
            x = input_img.view(input_img.size(0), -1)
        else:
            raise ValueError(f"Unexpected input_img shape: {input_img.shape}")
        
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0)
        elif len(condition.shape) == 2 and condition.shape[0] != input_img.shape[0]:
            raise ValueError(f"Batch size mismatch: condition {condition.shape}, input_img {input_img.shape}")
        
        x = torch.cat([condition, x], dim=1)
        return self.fc(x)

# === 2. 데이터 로딩 및 정규화 적용 ===
folder_path = '/mnt/c/Users/user/CGAN-PyTorch/comsol/data_comsol'
condition_vectors, input_matrices = process_all_files(folder_path)

input_matrices = np.array(input_matrices)  # (N, 56, 56)
condition_vectors = np.array(condition_vectors)

# --- 정규화 적용 (전체 데이터 기준으로) ---
input_matrices_norm, data_min, data_max = normalize_minmax(input_matrices)

# 텐서 변환
input_matrices = torch.tensor(input_matrices_norm, dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, 56, 56)
condition_vectors = torch.tensor(condition_vectors, dtype=torch.float32).to(device)               # (N, 7)

# === 모델 초기화 ===
generator = Generator(latent_dim, condition_dim).to(device)
discriminator = Discriminator(condition_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))

# === 로그, 샘플 저장용 디렉토리 생성 ===
log_file_path = './train_log.txt'
save_img_dir = './generated_samples'
os.makedirs(save_img_dir, exist_ok=True)

log_file = open(log_file_path, 'w')

# === 학습 루프 ===
num_epoch = 200  # 예시로 설정한 에폭 수
batch_size = 64  # 배치 크기 설정

for epoch in range(num_epoch):
    for i in tqdm(range(0, len(condition_vectors), batch_size)):

        batch_condition = condition_vectors[i:i+batch_size]  # (배치 크기, 7)
        batch_input = input_matrices[i:i+batch_size]  # (배치 크기, 1, 56, 56)

        real_img = batch_input.to(device)
        condition = batch_condition.to(device)

        current_batch_size = real_img.size(0)

        # Ground truth labels
        valid = torch.full((current_batch_size, 1), 1, dtype=torch.float32).to(device)
        fake = torch.full((current_batch_size, 1), 0, dtype=torch.float32).to(device)

        # === Generator 학습 ===
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        z = torch.randn(current_batch_size, latent_dim, device=device)

        try:
            gen_img = generator(z, condition)
            pred_fake = discriminator(condition, gen_img)
            loss_G = criterion(pred_fake, valid)
            if torch.isnan(loss_G):
                continue
            loss_G.backward()
            optimizer_G.step()
        except Exception as e:
            print(f"Generator step error: {e}")
            continue

        # === Discriminator 학습 ===
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        z = torch.randn(current_batch_size, latent_dim, device=device)

        try:
            pred_real = discriminator(condition, real_img)
            gen_img_detached = gen_img.detach()
            pred_fake = discriminator(condition, gen_img_detached)
            # 안정적 로그 확률 대신 BCEWithLogitsLoss 사용 권장. 원래 식은 log(pred_real) + log(1 - pred_fake)
            loss_D_real = criterion(pred_real, valid)
            loss_D_fake = criterion(pred_fake, fake)
            loss_D = (loss_D_real + loss_D_fake) / 2
            if torch.isnan(loss_D):
                continue
            loss_D.backward()
            optimizer_D.step()
        except Exception as e:
            print(f"Discriminator step error: {e}")
            continue

        # === 로그 저장 및 이미지 저장 ===
        if (i // batch_size) % 50 == 0:
            # 텍스트 로그 기록
            log_file.write(f"Epoch [{epoch+1}/{num_epoch}] Step [{i}/{len(condition_vectors)}] "
                           f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}\n")
            log_file.write(f"Discriminator Fake Output Sample: {pred_fake[:5].detach().cpu().numpy().flatten()}\n")
            log_file.write(f"Discriminator Real Output Sample: {pred_real[:5].detach().cpu().numpy().flatten()}\n")
            log_file.flush()

            print(f"Epoch [{epoch+1}/{num_epoch}] Step [{i}/{len(condition_vectors)}] "
                  f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

            # 이미지 저장 (PNG)
            save_image(gen_img, os.path.join(save_img_dir, f'gen_img_epoch{epoch+1}_step{i}.png'))

            # 필요하면 npy 저장 (주석 처리)
            # np.save(os.path.join(save_img_dir, f'gen_img_epoch{epoch+1}_step{i}.npy'), gen_img.detach().cpu().numpy())
            # np.save(os.path.join(save_img_dir, f'pred_fake_epoch{epoch+1}_step{i}.npy'), pred_fake.detach().cpu().numpy())
            # np.save(os.path.join(save_img_dir, f'pred_real_epoch{epoch+1}_step{i}.npy'), pred_real.detach().cpu().numpy())

    print(f"Epoch [{epoch+1}/{num_epoch}] 완료!")

log_file.close()

# === 인퍼런스 함수 및 저장 부분은 기존과 동일 ===
def generate_matrix_with_condition(generator, condition, latent_dim=100, device=None):
    z = torch.randn(1, latent_dim, device=device)
    condition_tensor = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_image = generator(z, condition_tensor)
    return generated_image.squeeze(0).cpu().numpy()

def save_generated_matrix_to_csv(matrix, output_path):
    if len(matrix.shape) == 4:
        matrix = matrix.squeeze(0).squeeze(0)
    elif len(matrix.shape) == 3:
        matrix = matrix.squeeze(0)

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.savetxt(output_path, matrix, delimiter=',')
    print(f"매트릭스가 {output_path}에 저장되었습니다.")

# === 예시 ===
condition_input = [60, 279, 2700, 0, 0, 25, 0]
generated_matrix = generate_matrix_with_condition(generator, condition_input, latent_dim, device)

output_file_path = '/mnt/c/Users/user/CGAN-PyTorch/result_comsol/generated_matrix.csv'
save_generated_matrix_to_csv(generated_matrix, output_file_path)

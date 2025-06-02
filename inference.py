import torch
from torchvision.utils import save_image
import os
import numpy as np
from main import Generator, normalize_cond_vec  # Generator 정의와 normalize 함수 가져오기
from tqdm import tqdm
import time

# 환경 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_name = "/home/rtxtitanx1/Seongmin/comsol/inference_png_colormap"
model_path = "/home/rtxtitanx1/Seongmin/comsol/result_png_colormap/best_model.pt"
noise_size = 100
condition_size = 7

os.makedirs(output_name, exist_ok=True)

# Generator 불러오기
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# 조건 벡터 리스트
c1 = 60
c2_list = list(range(300, 2701, 300))  # 300 ~ 2700 (9개)
c6_list = list(range(25, 181, 5))      # 25 ~ 180 (32개)

# 전체 조합 수
total_samples = len(c2_list) * len(c6_list)

# 시간 측정 시작
start_time = time.time()

# tqdm 진행 바 추가
sample_idx = 0
for c2 in tqdm(c2_list, desc="Generating images by condition c2"):
    c3 = 3000 - c2
    for c6 in c6_list:
        cond_vec = [c1, c2, c3, 0, 0, c6, 0]
        cond_tensor = torch.tensor(cond_vec, dtype=torch.float32)
        cond_tensor = normalize_cond_vec(cond_tensor).unsqueeze(0).to(device)

        noise = torch.randn(1, noise_size).to(device)

        with torch.no_grad():
            fake_img = generator(noise, cond_tensor)

        save_name = f"infer_{c1}_{c2}_{c3}_0_0_{c6}_0.png"
        save_path = os.path.join(output_name, save_name)
        save_image(fake_img, save_path, normalize=True)

        sample_idx += 1

# 시간 측정 종료
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n✅ 총 {sample_idx}개의 이미지 생성 완료.")
print(f"🕒 총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time / 60:.2f}분)")

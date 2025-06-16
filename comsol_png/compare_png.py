import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import lpips
import subprocess
import tempfile

# 폴더 경로 지정
real_dir = '/home/gtx2080tix3/seongmin/temp/comsol/data_png_colormap'
gen_dir = '/home/gtx2080tix3/seongmin/temp/comsol/inference_png_colormap'

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# 파일명 딕셔너리 생성
real_files = {f: os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
gen_files = {f.replace('infer_', ''): os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}

# 공통 key (원본 파일명 기준) 정렬
common_keys = sorted(set(real_files.keys()) & set(gen_files.keys()))

# PSNR, SSIM, LPIPS 계산 준비
psnr_list, ssim_list, lpips_list = [], [], []
lpips_model = lpips.LPIPS(net='alex')

for fname in tqdm(common_keys, desc='Computing PSNR/SSIM/LPIPS'):
    real_path = real_files[fname]
    gen_path = gen_files[fname]
    
    real_img = Image.open(real_path).convert('RGB')
    gen_img = Image.open(gen_path).convert('RGB')
    
    real_arr = np.array(real_img)
    gen_arr = np.array(gen_img)

    psnr_list.append(psnr(real_arr, gen_arr, data_range=255))
    ssim_list.append(ssim(real_arr, gen_arr, channel_axis=2, data_range=255))  # 여기만 수정

    # LPIPS 계산
    real_tensor = transforms.ToTensor()(real_img).unsqueeze(0)
    gen_tensor = transforms.ToTensor()(gen_img).unsqueeze(0)
    lpips_dist = lpips_model(real_tensor, gen_tensor)
    lpips_list.append(lpips_dist.item())

# FID 계산 (pytorch-fid 이용, 임시 폴더 복사)
with tempfile.TemporaryDirectory() as tmp_real, tempfile.TemporaryDirectory() as tmp_gen:
    for fname in common_keys:
        Image.open(real_files[fname]).save(os.path.join(tmp_real, fname))
        Image.open(gen_files[fname]).save(os.path.join(tmp_gen, fname))

    print("Calculating FID...")
    subprocess.run(['python', '-m', 'pytorch_fid', tmp_real, tmp_gen])

# IS 계산 함수
def inception_score(imgs, splits=10):
    model = inception_v3(pretrained=True, transform_input=False).eval().cuda()
    up = transforms.Resize((299, 299))
    imgs = torch.stack([up(img) for img in imgs]).cuda()
    with torch.no_grad():
        preds = model(imgs)
    preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
    split_scores = []
    for k in range(splits):
        part = preds[k * len(preds) // splits: (k+1) * len(preds) // splits]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        split_scores.append(np.exp(kl))
    return np.mean(split_scores), np.std(split_scores)

img_tensors = []
for fname in tqdm(common_keys, desc="Loading images for IS"):
    gen_img = Image.open(gen_files[fname]).convert('RGB')
    img_tensors.append(transform(gen_img))

print("Calculating Inception Score...")
mean_is, std_is = inception_score(img_tensors)
print(f'Inception Score: {mean_is:.4f} ± {std_is:.4f}')

# Accuracy / Precision / Recall 계산 (SSIM threshold 기반)
threshold = 0.90
similarity_flags = [s > threshold for s in ssim_list]
gt = [1]*len(similarity_flags)  # 모두 정답(True)이라고 가정
accuracy = accuracy_score(gt, similarity_flags)
precision = precision_score(gt, similarity_flags)
recall = recall_score(gt, similarity_flags)

# 최종 결과 출력
print(f"\n=== Evaluation Results ===")
print(f"PSNR:  {np.mean(psnr_list):.4f}")
print(f"SSIM:  {np.mean(ssim_list):.4f}")
print(f"LPIPS: {np.mean(lpips_list):.4f}")
print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
print(f"Accuracy (SSIM>{threshold}): {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

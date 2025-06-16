import os
import re
import matplotlib.pyplot as plt

# 🔸 사용자가 직접 지정해야 할 부분
log_files = [
    "/home/gtx2080tix3/seongmin/comsol/result_png_colormap_final_origin/full_log.txt",
    "/home/gtx2080tix3/seongmin/comsol/result_png_colormap_final_lable/full_log.txt",
    "/home/gtx2080tix3/seongmin/comsol/result_png_colormap_final_lowdlr/full_log.txt",
    "/home/gtx2080tix3/seongmin/comsol/result_png_colormap_final_smallnoise/full_log.txt",
    "/home/gtx2080tix3/seongmin/comsol/result_png_colormap_final_trainstep/full_log.txt"
]

labels = [
    "Original",
    "Label Smoothing",
    "Low Discriminator Learning rate",
    "Add small noise",
    "Train step manipulation"
]

save_dir = "/home/gtx2080tix3/seongmin/comsol"  # 저장할 폴더 경로 (예: "./plots")

# 🔸 저장 디렉토리 생성
os.makedirs(save_dir, exist_ok=True)

# 색상 설정 (4개)
colors = ['r', 'g', 'b', 'orange', 'purple']

# 데이터 저장 리스트
ssim_data = []
lpips_data = []

# 로그 파싱
for file in log_files:
    ssim_list = []
    lpips_list = []
    with open(file, 'r') as f:
        for line in f:
            match = re.match(r"Epoch\s+(\d+):\s+SSIM=([0-9.]+),\s+LPIPS=([0-9.]+)", line)
            if match:
                epoch, ssim, lpips_val = match.groups()
                ssim_list.append(float(ssim))
                lpips_list.append(float(lpips_val))
    ssim_data.append(ssim_list)
    lpips_data.append(lpips_list)

# 🔹 SSIM Plot
plt.figure(figsize=(10, 6))
for i in range(len(ssim_data)):
    plt.plot(range(1, len(ssim_data[i]) + 1), ssim_data[i], label=labels[i], color=colors[i])
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("SSIM over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ssim_comparison.png"))
plt.close()

# 🔹 LPIPS Plot
plt.figure(figsize=(10, 6))
for i in range(len(lpips_data)):
    plt.plot(range(1, len(lpips_data[i]) + 1), lpips_data[i], label=labels[i], color=colors[i])
plt.xlabel("Epoch")
plt.ylabel("LPIPS")
plt.title("LPIPS over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lpips_comparison.png"))
plt.close()

print(f"그래프 저장 완료: {save_dir}")

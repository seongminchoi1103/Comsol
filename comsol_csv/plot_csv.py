import matplotlib.pyplot as plt
import re
import os

# 로그 파일 경로와 저장할 이미지 경로
log_file = '/home/gtx2080tix3/seongmin/comsol/result_csvmtx_4/full_log.txt'
save_path = '/home/gtx2080tix3/seongmin/comsol/result_csvmtx_4/results/loss_plot.jpg'  # <- 여기서 원하는 경로로 수정하세요

# 디렉토리 없으면 생성
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 에폭, R2, MAE 저장용 리스트
epochs = []
r2_scores = []
mae_scores = []

# 로그 파일 읽기 및 파싱
with open(log_file, 'r') as f:
    for line in f:
        match = re.match(r"Epoch (\d+): R2=([-\d.]+), MAE=([-\d.]+)", line.strip())
        if match:
            epoch = int(match.group(1))
            r2 = float(match.group(2))
            mae = float(match.group(3))
            epochs.append(epoch)
            r2_scores.append(r2)
            mae_scores.append(mae)

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 5))

# R² 플롯 (왼쪽 y축)
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('R² Score', color=color)
ax1.plot(epochs, r2_scores, color=color, label='R² Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# MAE 플롯 (오른쪽 y축)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('MAE', color=color)
ax2.plot(epochs, mae_scores, color=color, linestyle='--', label='MAE')
ax2.tick_params(axis='y', labelcolor=color)

# 제목 및 저장
plt.title('R² Score and MAE over Epochs')
fig.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"그래프가 다음 경로에 저장되었습니다: {save_path}")

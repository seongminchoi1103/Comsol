import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

input_dir = '/home/gtx2080tix3/seongmin/comsol/data/data_comsol'
output_dir = '/home/gtx2080tix3/seongmin/comsol/data/data_png_colormap'
os.makedirs(output_dir, exist_ok=True)

# 사용할 컬러맵 지정: inferno, viridis, plasma, jet 등 사용 가능
colormap = cm.get_cmap('inferno')

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_dir, filename)

        # CSV 로드 후 1행 1열 제거 및 float 변환
        data = pd.read_csv(csv_path, header=None).values.astype(float)

        # shape 확인
        assert data.shape == (56, 56), f"{filename}의 shape가 56x56이 아닙니다: {data.shape}"

        # 0~1 정규화
        data_norm = data / np.max(data) if data.max() > 0 else data

        # 컬러맵 적용 후 RGBA → RGB 변환
        color_img = colormap(data_norm)[:, :, :3]  # R, G, B만 사용
        color_img = (color_img * 255).astype(np.uint8)

        # 이미지로 저장
        img = Image.fromarray(color_img)
        png_filename = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join(output_dir, png_filename))

print("컬러맵을 적용하여 모든 CSV 파일이 컬러 PNG로 저장되었습니다.")

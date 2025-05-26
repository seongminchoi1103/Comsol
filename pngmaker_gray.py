import os
import pandas as pd
import numpy as np
from PIL import Image

input_dir = '/mnt/c/Users/user/CGAN-PyTorch/comsol/data_comsol'
output_dir = '/mnt/c/Users/user/CGAN-PyTorch/comsol/data_png'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_dir, filename)

        # CSV 로드 후 1행 1열 제거 및 float 변환
        data = pd.read_csv(csv_path, header=None).values[1:, 1:].astype(float)

        # NaN 값이 있다면 0으로 대체
        data = np.nan_to_num(data, nan=0.0)

        # shape 확인 (56, 56)
        assert data.shape == (56, 56), f"{filename}의 shape가 56x56이 아닙니다: {data.shape}"

        # 0~255 정규화 (데이터가 0~1 범위일 경우)
        if data.max() <= 1.0:
            data = (data * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

        # 이미지로 변환 및 저장
        img = Image.fromarray(data, mode='L')
        png_filename = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join(output_dir, png_filename))

print("모든 CSV 파일이 PNG로 변환 완료되었습니다. (NaN 값은 0으로 처리됨)")

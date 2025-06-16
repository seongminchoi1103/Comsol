import os
import pandas as pd
import numpy as np
from io import StringIO

# === 1. 설정 ===
INPUT_CSV = "2waveguide_288_56x56.csv"         # 입력 파일명
OUTPUT_DIR = "data_comsol"                      # 결과 저장 폴더

# === 2. CSV 파일 읽기 ===
with open(INPUT_CSV, 'r') as f:
    lines = f.readlines()

# 헤더 처리: 9번째 줄이 진짜 헤더
raw_header = lines[8].strip().split(',')
new_header = ['x', 'y']

for i in range(2, len(raw_header), 3):
    if i + 2 < len(raw_header):
        t_str = raw_header[i].strip()
        p_str = raw_header[i + 1].strip()
        phi_str = raw_header[i + 2].strip()

        t = int(t_str.split('=')[-1])
        p1 = int(p_str.split('=')[-1])
        p2 = 3000 - p1
        p3 = 0
        phi2 = int(phi_str.split('=')[-1])
        phi1 = 0
        phi3 = 0

        vec = [t, p1, p2, p3, phi1, phi2, phi3]
        new_header.append(str(vec))

# 데이터 부분 읽기
data_str = '\n'.join(lines[9:])
df = pd.read_csv(StringIO(data_str), header=None)
df.columns = new_header

# === 3. condition matrix 저장 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

x_unique = np.sort(df['x'].unique())
y_unique = np.sort(df['y'].unique())

skipped = 0
created = 0

for col in df.columns[2:]:  # 첫 2개(x, y)는 제외
    # 안전한 파일명 생성
    safe_name = col.replace('[', '').replace(']', '').replace(', ', '_').replace(',', '_').replace(' ', '')
    file_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")

    # 이미 파일이 존재하면 건너뜀
    if os.path.exists(file_path):
        print(f"⚠️ 이미 존재함, 생략: {file_path}")
        skipped += 1
        continue

    # matrix 생성
    matrix = np.full((len(y_unique), len(x_unique)), np.nan)
    for _, row in df.iterrows():
        x_idx = np.where(x_unique == row['x'])[0][0]
        y_idx = np.where(y_unique == row['y'])[0][0]
        matrix[y_idx, x_idx] = row[col]

    mat_df = pd.DataFrame(matrix, index=y_unique, columns=x_unique)
    mat_df.to_csv(file_path, index_label='y')
    print(f"✅ 생성됨: {file_path}")
    created += 1

print(f"\n🎯 완료: 생성된 파일 {created}개, 생략된 파일 {skipped}개")

# === 4. 후처리: NaN → 0, 첫 행/열 제거 및 덮어쓰기 ===

for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith('.csv'):
        filepath = os.path.join(OUTPUT_DIR, filename)
        data = pd.read_csv(filepath, index_col=0).values  # y컬럼 인덱스 인식

        # NaN → 0
        data = np.nan_to_num(data, nan=0.0)

        # 덮어쓰기 저장
        pd.DataFrame(data).to_csv(filepath, header=False, index=False)

print(f"🎯 후처리 완료: {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])}개 파일 덮어쓰기 완료")

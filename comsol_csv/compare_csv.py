import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

folder1 = '/home/gtx2080tix3/seongmin/comsol/data_comsol_processed'
folder2 = '/home/gtx2080tix3/seongmin/comsol/result_generated_csv_2'

mae_list = []
rmse_list = []
r2_list = []

for filename in sorted(os.listdir(folder1)):
    if filename.endswith('.csv'):
        key = filename  # 예: "60_300_2700_0_0_25_0.csv"
        gen_name = f"infer_{key}"  # 예: "infer_60_300_2700_0_0_25_0.csv"

        path1 = os.path.join(folder1, key)
        path2 = os.path.join(folder2, gen_name)
        
        if not os.path.exists(path2):
            print(f"{gen_name}가 {folder2}에 없습니다. 건너뜁니다.")
            continue

        data1 = pd.read_csv(path1, header=None).values.astype(float)
        data2 = pd.read_csv(path2, header=None).values.astype(float)

        # shape 검증
        if data1.shape != (56, 56) or data2.shape != (56, 56):
            print(f"{filename}의 shape가 맞지 않습니다: {data1.shape} vs {data2.shape}. 건너뜁니다.")
            continue

        # Flatten for metric calculation
        y_true = data1.flatten()
        y_pred = data2.flatten()

        # MAE / RMSE / R² 계산
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

# 평균 결과 출력
if mae_list and rmse_list and r2_list:
    print(f"\n✅ 총 {len(mae_list)}개 파일 비교 완료")
    print(f"📈 평균 MAE: {np.mean(mae_list):.6f}")
    print(f"📉 평균 RMSE: {np.mean(rmse_list):.6f}")
    print(f"📊 평균 R²: {np.mean(r2_list):.6f}")
else:
    print("비교할 데이터가 없습니다.")

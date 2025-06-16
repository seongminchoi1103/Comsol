import re
from PIL import Image
import os

img_folder = "/mnt/c/Users/user/CGAN-PyTorch/comsol/result_png_colormap_2"
gif_path = "/mnt/c/Users/user/CGAN-PyTorch/comsol/result_colormap2.gif"

# epoch 번호 추출하는 함수
def extract_epoch_num(filename):
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else -1

# PNG 파일만 필터링하고, epoch 숫자 기준 정렬
png_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]
png_files.sort(key=extract_epoch_num)

imgs = [Image.open(os.path.join(img_folder, f)) for f in png_files]

imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)

import os
from torchvision.utils import save_image
from PIL import Image

output_name = "/mnt/c/Users/user/CGAN-PyTorch/comsol/result_png_colormap"

os.makedirs(output_name, exist_ok=True)

def save_ground_truth_samples(gt_folder, save_path, num_samples=32, img_size=56):
    import os
    from torchvision.utils import save_image
    import torch
    from torchvision import transforms

    file_names = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])[:num_samples]
    gt_images = []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # (C, H, W), [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # generator와 동일하게 정규화
    ])

    for fn in file_names:
        img_path = os.path.join(gt_folder, fn)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        gt_images.append(img)

    gt_images_tensor = torch.stack(gt_images)  # (num_samples, 3, 56, 56)
    save_image(gt_images_tensor, save_path, nrow=8)
    print(f"Ground truth samples saved to {save_path}")

save_ground_truth_samples('/mnt/c/Users/user/CGAN-PyTorch/comsol/data_png_colormap', os.path.join(output_name, 'ground_truth_samples.png'))
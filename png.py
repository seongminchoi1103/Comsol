import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Hyper-parameters
num_epoch = 1000
batch_size = 32
learning_rate = 0.0001
num_channel = 3  # RGB 컬러 이미지
img_size = 56 * 56 * num_channel
dir_name = "/mnt/c/Users/user/CGAN-PyTorch/comsol/data_png_colormap"
output_name = "/mnt/c/Users/user/CGAN-PyTorch/comsol/result_png_colormap"

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

condition_size = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} device".format(device))

os.makedirs(output_name, exist_ok=True)
os.makedirs(dir_name, exist_ok=True)

def normalize_cond_vec(cond_vec):
    cond_vec[0] = 1.0
    cond_vec[1:4] = cond_vec[1:4] / 3000.0
    cond_vec[4:7] = cond_vec[4:7] / 180.0
    return cond_vec

# --- 수정된 Dataset ---
class CustomPNGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.data_dir, file_name)

        image = Image.open(img_path).convert('RGB')  # <-- 컬러 PNG
        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(file_name)[0]
        cond_strs = base_name.split('_')
        assert len(cond_strs) == condition_size, f"Filename condition vector size mismatch: {file_name}"

        cond_vec = torch.tensor([float(c) for c in cond_strs], dtype=torch.float32)
        cond_vec = normalize_cond_vec(cond_vec)
        return image, cond_vec

# --- 수정된 Transform ---
transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # RGB
])

custom_dataset = CustomPNGDataset(dir_name, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(img_size + condition_size, hidden_size3)
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(noise_size + condition_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x

def check_condition(_generator):
    _generator.eval()
    test_images = []

    cond_vectors = [
        [60, 2700, 300, 0, 0, 180, 0],
        [60, 1500, 600, 0, 0, 90, 0],
        [60, 3000, 450, 0, 0, 45, 0],
    ]

    for cond_vec_list in cond_vectors:
        cond_vec = torch.tensor(cond_vec_list, dtype=torch.float32).unsqueeze(0).to(device)
        cond_vec = normalize_cond_vec(cond_vec.squeeze()).unsqueeze(0).to(device)
        z = torch.randn(1, noise_size).to(device)
        z_cond = torch.cat([z, cond_vec], dim=1)
        with torch.no_grad():
            fake_img = _generator(z_cond)
        test_images.append(fake_img)

    test_images = torch.cat(test_images, dim=0)
    test_images = test_images.reshape(-1, 3, 56, 56)
    save_image(test_images, os.path.join(output_name, 'CGAN_test_result.png'), nrow=10)
    _generator.train()

discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epoch}]")
    for i, (images, cond_vec) in enumerate(loop):

        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # RGB 이미지를 흑백처럼 다루기 위해 mean 연산 또는 1채널로 압축할 수도 있음.
        # 하지만 지금은 그냥 reshape해서 사용 (3채널 → flatten해서 무시됨)
        images = images.view(batch_size, -1).to(device)
        real_input = torch.cat([images, cond_vec.to(device)], dim=1)

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        z = torch.randn(batch_size, noise_size).to(device)
        z_cond = torch.cat([z, cond_vec.to(device)], dim=1)

        fake_images = generator(z_cond)
        fake_input = torch.cat([fake_images, cond_vec.to(device)], dim=1)

        g_loss = criterion(discriminator(fake_input), real_label)
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        z = torch.randn(batch_size, noise_size).to(device)
        z_cond = torch.cat([z, cond_vec.to(device)], dim=1)
        fake_images = generator(z_cond)
        fake_input = torch.cat([fake_images, cond_vec.to(device)], dim=1)

        fake_loss = criterion(discriminator(fake_input), fake_label)
        real_loss = criterion(discriminator(real_input), real_label)
        d_loss = (fake_loss + real_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

    samples = fake_images.view(batch_size, 3, 56, 56)
    save_image(samples, os.path.join(output_name, f'CGAN_fake_samples_epoch{epoch+1}.png'))

    fake_img = fake_images[0].detach().cpu().numpy()  # shape: (9408,)
    fake_img = fake_img.reshape(3, 56, 56)            # (3, 56, 56)
    fake_img = fake_img.transpose(1, 2, 0)            # (56, 56, 3)

    fake_img = (fake_img + 1) / 2                      # [-1,1] -> [0,1]
    fake_img = (fake_img * 255).clip(0, 255).astype('uint8')

check_condition(generator)

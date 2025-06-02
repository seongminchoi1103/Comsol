import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Hyper-parameters
num_epoch = 1500
batch_size = 32
learning_rate = 0.0001
img_channels = 3
img_size = 56
condition_size = 7
noise_size = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} device".format(device))

dir_name = "/home/rtxtitanx1/Seongmin/comsol/data_png_colormap"
output_name = "/home/rtxtitanx1/Seongmin/comsol/result_png_colormap"
os.makedirs(output_name, exist_ok=True)

def normalize_cond_vec(cond_vec):
    cond_vec[0] = 1.0
    cond_vec[1:4] = cond_vec[1:4] / 3000.0
    cond_vec[4:7] = cond_vec[4:7] / 180.0
    return cond_vec

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

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(file_name)[0]
        cond_strs = base_name.split('_')
        assert len(cond_strs) == condition_size, f"Filename condition vector size mismatch: {file_name}"

        cond_vec = torch.tensor([float(c) for c in cond_strs], dtype=torch.float32)
        cond_vec = normalize_cond_vec(cond_vec)
        return image, cond_vec

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

custom_dataset = CustomPNGDataset(dir_name, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# --- Generator with CNN transpose layers ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(noise_size + condition_size, 256 * 7 * 7)
        self.deconv_layers = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 28x28 -> 56x56
            nn.Tanh()
        )

    def forward(self, noise, cond_vec):
        x = torch.cat([noise, cond_vec], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.deconv_layers(x)
        return x

# --- Discriminator with CNN ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 이미지 입력 + condition 벡터를 채널에 맞춰 concat: 이미지 3채널 + 1 채널 condition 확장
        # condition vector를 공간 차원(56x56)에 맞게 반복 확장해서 이미지와 concat
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_size, img_size * img_size),
            nn.LeakyReLU(0.2),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 4, 2, 1),  # 56x56 -> 28x28
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),               # 28x28 -> 14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),              # 14x14 -> 7x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond_vec):
        # cond_vec: (B, condition_size) -> (B, 1, 56, 56)
        cond_map = self.condition_embed(cond_vec)
        cond_map = cond_map.view(-1, 1, img_size, img_size)
        x = torch.cat([img, cond_map], dim=1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

def check_condition(_generator):
    _generator.eval()
    cond_vectors = [
        [60, 2700, 300, 0, 0, 180, 0],
        [60, 1500, 600, 0, 0, 90, 0],
        [60, 3000, 450, 0, 0, 45, 0],
    ]
    cond_tensors = []
    for cond_vec_list in cond_vectors:
        cond_vec = torch.tensor(cond_vec_list, dtype=torch.float32)
        cond_vec = normalize_cond_vec(cond_vec).unsqueeze(0).to(device)
        cond_tensors.append(cond_vec)
    cond_batch = torch.cat(cond_tensors, dim=0)
    noise = torch.randn(len(cond_vectors), noise_size).to(device)
    with torch.no_grad():
        fake_images = _generator(noise, cond_batch)
    save_image(fake_images, os.path.join(output_name, 'CGAN_test_result.png'), normalize=True)
    _generator.train()

# --- 학습 루프 ---
def train():
    for epoch in range(num_epoch):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epoch}]")
        for images, cond_vec in loop:
            batch_size = images.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            images = images.to(device)
            cond_vec = cond_vec.to(device)

            # Discriminator 학습
            d_optimizer.zero_grad()
            d_real_pred = discriminator(images, cond_vec)
            d_real_loss = criterion(d_real_pred, real_label)

            noise = torch.randn(batch_size, noise_size).to(device)
            fake_images = generator(noise, cond_vec)

            d_fake_pred = discriminator(fake_images.detach(), cond_vec)
            d_fake_loss = criterion(d_fake_pred, fake_label)

            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            # Generator 학습
            g_optimizer.zero_grad()
            g_fake_pred = discriminator(fake_images, cond_vec)
            g_loss = criterion(g_fake_pred, real_label)
            g_loss.backward()
            g_optimizer.step()

            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        # Epoch마다 결과 저장
        with torch.no_grad():
            test_noise = torch.randn(batch_size, noise_size).to(device)
        test_cond = cond_vec
        fake_samples = generator(test_noise, test_cond)

        # real_images와 fake_samples를 가로로 붙이기 (dim=3: width 방향)
        comparison = torch.cat([images.cpu(), fake_samples.cpu()], dim=3)  # detach()해도 됨

        # 저장 (normalize=True 유지)
        save_image(comparison, os.path.join(output_name, f'CGAN_compare_epoch{epoch+1}.png'), normalize=True)
        
        if epoch == num_epoch - 1:
            torch.save(generator.state_dict(), os.path.join(output_name, 'best_model.pt'))

if __name__ == "__main__":
    train()
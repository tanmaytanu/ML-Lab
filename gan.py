# প্রয়োজনীয় লাইব্রেরি ইমপোর্ট করা
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ডিভাইস সেট করা: যদি CUDA (GPU) থাকে তাহলে GPU ব্যবহার করবে, না হলে CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ইমেজকে টেনসরে রূপান্তর করা এবং নরমালাইজ করা [-1, 1] রেঞ্জে
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 ডেটাসেট ডাউনলোড এবং লোড করা
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# হাইপারপ্যারামিটার নির্ধারণ
latent_dim = 100          # ল্যাটেন্ট স্পেসের সাইজ
lr = 0.0002               # লার্নিং রেট
beta1 = 0.5               # Adam optimizer এর বিটা ১
beta2 = 0.999             # Adam optimizer এর বিটা ২
num_epochs = 10           # ট্রেইনিংয়ের মোট ইপোক সংখ্যা

# জেনারেটর ক্লাস ডিফাইন করা
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),  # ল্যাটেন্ট ভেক্টরকে ফিচার ম্যাপে রূপান্তর
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),        # 2D আকারে রূপান্তর
            nn.Upsample(scale_factor=2),         # ৮x৮ -> ১৬x১৬
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),         # ১৬x১৬ -> ৩২x৩২
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Output RGB image
            nn.Tanh()  # আউটপুট স্কেল [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img

# ডিসক্রিমিনেটর ক্লাস ডিফাইন করা
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid()  # প্রোবাবিলিটি রিটার্ন করে (fake or real)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# জেনারেটর এবং ডিসক্রিমিনেটর ইনিশিয়ালাইজ করা
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# লস ফাংশন: Binary Cross Entropy
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# ট্রেইনিং লুপ শুরু
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        real_images = batch[0].to(device)

        # ট্রু (১) এবং ফেইক (০) লেবেল তৈরি
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)

        # ---------------------
        # 1. ডিসক্রিমিনেটর ট্রেইন
        # ---------------------
        optimizer_D.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim, device=device)
        fake_images = generator(z)

        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 2. জেনারেটর ট্রেইন
        # ---------------------
        optimizer_G.zero_grad()
        gen_images = generator(z)
        g_loss = adversarial_loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()

        # প্রতি ১০০ ব্যাচে একবার প্রগ্রেস প্রিন্ট করে
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )

    # প্রতি ১০ ইপোকে জেনারেটেড ইমেজ দেখানো
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()


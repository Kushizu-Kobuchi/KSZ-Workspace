# ===
# 导包
# ===

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchkeras import summary
from torchvision import transforms, datasets

from xyz_util import *

# ===
# 配置及超参数
# ===

fix_seed(42)

READ_MODEL_PATH = False
SAVE_MODEL_PATH = './model/autoencoder_MNIST.pth'
FORCE_CPU = False
device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"

N_EPOCH = 50
BATCH_SIZE = 128

lr = 1e-2
weight_decay = 1e-5

# ===
# 加载数据集及数据预处理
# ===

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_data = datasets.MNIST(
    root='./data', train=True, transform=data_transform, download=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)


# ===
# 模型定义及实例化
# ===

class MyAutoEncoder(nn.Module):
    def __init__(self):
        super(MyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


model = MyAutoEncoder()
summary(model, input_shape=(784,))
model.to(device)

if READ_MODEL_PATH:
    model.load_state_dict(torch.load(READ_MODEL_PATH))
else:
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

# ===
# 训练
# ===

    for epoch in range(N_EPOCH):
        if epoch in [N_EPOCH * 0.25, epoch * 0.5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        for img_batch, _ in train_loader:
            img_batch: torch.Tensor
            img_batch = img_batch.to(device)
            img_batch = img_batch.view(img_batch.size(0), -1)

            _, output = model(img_batch)  # 预测
            loss: torch.Tensor = loss_fn(output, img_batch)  # 损失
            optimizer.zero_grad()  # 清零
            loss.backward()  # 反向
            optimizer.step()  # 传播
        print(f'epoch={epoch}, loss={loss}')

    if SAVE_MODEL_PATH:
        torch.save(model.state_dict(), SAVE_MODEL_PATH)


# ===
# 其他
# ===

def to_img(x: torch.Tensor) -> torch.Tensor:
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


test_code = torch.FloatTensor([[1.19, -3.36, 2.06]]).to(device)
test_decode = model.decoder(test_code)
decode_img = to_img(test_decode).squeeze()
decode_img = decode_img.data.cpu().numpy() * 255
plt.imshow(decode_img.astype('uint8'), cmap='gray')
plt.show()

print("运行完毕")

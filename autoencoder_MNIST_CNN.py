# ===
# 导包
# ===

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchkeras import summary
from torchvision import transforms, datasets

from xyz_util import *

THIS_FILE_NAME = os.path.basename(__file__).strip('.py')

# ===
# 配置及超参数
# ===

fix_seed(42)

READ_MODEL_PATH = f'./model/{THIS_FILE_NAME}__epoch100.pth'
SAVE_MODEL_PATH = f'./model/{THIS_FILE_NAME}__epoch100.pth'
READ_MODEL_PATH = False
# SAVE_MODEL_PATH = False
FORCE_CPU = False
device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"

N_EPOCH = 2
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
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


model = MyAutoEncoder()
summary(model, input_shape=(1, 28, 28))
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
    loss_epochs = []
    for epoch in range(N_EPOCH):
        if epoch in [N_EPOCH * 0.25, epoch * 0.5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        loss_in_epoch = []

        for img_batch, _ in train_loader:
            img_batch: torch.Tensor
            img_batch = img_batch.to(device)
            # DEL: img_batch = img_batch.view(img_batch.size(0), -1)

            _, output = model(img_batch)  # 预测
            loss: torch.Tensor = loss_fn(output, img_batch)  # 损失
            optimizer.zero_grad()  # 清零
            loss.backward()  # 反向
            optimizer.step()  # 传播

            loss_in_epoch.append(loss.data.cpu())

        print(f'epoch={epoch}, loss={np.mean(loss_in_epoch)}')
        loss_epochs.append(np.mean(loss_in_epoch))

    if SAVE_MODEL_PATH:
        torch.save(model.state_dict(), SAVE_MODEL_PATH)

# ===
# 运行时数据保存
# ===

fig = plt.figure()
plt.plot(loss_epochs)
plt.show()
fig.savefig(f'./runtime/loss_epochs/{THIS_FILE_NAME}.png')

# ===
# 其他
# ===

def to_img(x: torch.Tensor) -> torch.Tensor:
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


for img_batch, _ in train_loader:
    img = img_batch[0]
    img = to_img(img).squeeze()
    img = img.data.numpy() * 255
    plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()
    img_batch = img_batch.to(device)
    _, decode_imgs = model(img_batch)
    decode_img = to_img(decode_imgs).squeeze()
    decode_img = decode_img.data.cpu().numpy() * 255
    plt.imshow(decode_img[0].astype('uint8'), cmap='gray')
    plt.show()
    break


print("运行完毕")

# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
# https://blog.csdn.net/sikh_0529/article/details/127818626

# ===
# 导包
# ===

import copy

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchkeras import summary

from xyz_util import *

THIS_FILE_NAME = os.path.basename(__file__).strip('.py')

# ===
# 配置及超参数
# ===

RANDOM_SEED = 42
fix_seed(RANDOM_SEED)

READ_MODEL_PATH = f'./model/{THIS_FILE_NAME}.pth'
SAVE_MODEL_PATH = f'./model/{THIS_FILE_NAME}.pth'
if True:
    READ_MODEL_PATH = False
else:
    SAVE_MODEL_PATH = False
FORCE_CPU = False
device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"

N_EPOCH = 2
# BATCH_SIZE = 128

lr = 1e-3

# ===
# 加载数据集及数据预处理
# ===

# 500行 141列 第一列是标签 标签为1是正常 2345是异常
train_data = pd.read_csv(
    './data/ECG5000/ECG5000_TRAIN.tsv', sep='\t', header=None)
# 4500行 ...
test_data = pd.read_csv(
    './data/ECG5000/ECG5000_TEST.tsv', sep='\t', header=None)

# plt.plot(train_data.transpose()[0][1:])
# plt.show()

data: pd.DataFrame = pd.concat([train_data, test_data], axis=0)
data = data.sample(frac=1.0)

new_columns = list(data.columns)
new_columns[0] = 'target'
data.columns = new_columns

normal_data = data[data.target == 1].drop(labels='target', axis=1)
anomaly_data = data[data.target != 1].drop(labels='target', axis=1)

train_data, val_data = train_test_split(
    normal_data,
    test_size=0.15,
    random_state=RANDOM_SEED)

val_data, test_data = train_test_split(
    val_data,
    test_size=0.33,
    random_state=RANDOM_SEED)


def create_dataset(df: pd.DataFrame):
    """
    DataFrame转list(Tensor)
    """
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


# 数据集 序列长140 特征数1
train_dataset, seq_len, n_features = create_dataset(train_data)
val_dataset, _, _ = create_dataset(val_data)
test_normal_dataset, _, _ = create_dataset(test_data)
test_anomaly_dataset, _, _ = create_dataset(anomaly_data)


# ===
# 模型定义及实例化
# ===


class MyLSTMAutoEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128):
        super(MyLSTMAutoEncoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim
        # n_features -> hidden_dim
        self.encoder_rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # hidden_dim -> embedding_dim
        self.encoder_rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        # embedding_dim -> embedding_dim
        self.decoder_rnn1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        # hidden_dim -> embedding_dim
        self.decoder_rnn2 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # embedding_dim -> n_features
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        # 编码器
        # (140, 1) -> (1, 140, 1)
        x = x.reshape((1, self.seq_len, self.n_features))
        # (1, 140, 256) hidden(1, 1, 256) cell(1, 1, 256)
        x, (_, _) = self.encoder_rnn1(x)
        _, (hidden_n, _) = self.encoder_rnn2(x)  # hidden(1, 1, 128)
        x = hidden_n.reshape((self.n_features, self.embedding_dim))  # (1, 128)

        # 译码器
        x = x.repeat(self.seq_len, self.n_features)  # (140, 128)
        x = x.reshape((self.n_features, self.seq_len,
                       self.embedding_dim))  # (1, 140, 128)
        x, (_, _) = self.decoder_rnn1(x)  # (1, 140, 128)
        x, (_, _) = self.decoder_rnn2(x)  # (1, 140, 256)
        x = x.reshape((self.seq_len, self.hidden_dim))  # (140, 256)
        x = self.output_layer(x)  # (140, 1)
        return x


model = MyLSTMAutoEncoder(
    seq_len=seq_len, n_features=n_features, embedding_dim=128)
summary(model, input_data=train_dataset[0])
model.to(device)

if READ_MODEL_PATH:
    model.load_state_dict(torch.load(READ_MODEL_PATH))
    # model.load_state_dict(load_model_from_json(f'./model/{THIS_FILE_NAME}.json'))
else:
    loss_fn = nn.L1Loss(reduction='sum')
    loss_fn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ===
    # 训练
    # ===
    print('开始训练')
    history = dict(train=[], val=[])  # loss记录
    best_model_dict = copy.deepcopy(model.state_dict())
    best_loss_val = 10000.

    for epoch in range(N_EPOCH):
        loss_train = []
        model = model.train()
        for train_seq in train_dataset:
            train_seq = train_seq.to(device)

            pred = model(train_seq)
            loss = loss_fn(pred, train_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

        loss_val = []
        model.eval()
        with torch.no_grad():
            for val_seq in val_dataset:
                val_seq = val_seq.to(device)
                pred = model(pred)
                loss = loss_fn(pred, val_seq)

                loss_val.append(loss.item())

        loss_mean_train = np.mean(loss_train)
        loss_mean_val = np.mean(loss_val)
        history['train'].append(loss_mean_train)
        history['val'].append(loss_mean_val)

        if loss_mean_val < best_loss_val:
            best_loss_val = loss_mean_val
            best_model_dict = copy.deepcopy(model.state_dict())

        print(
            f'epoch={epoch} trainloss={loss_mean_train} valloss={loss_mean_val}')
    print('训练结束')

    model.load_state_dict(best_model_dict)
    if SAVE_MODEL_PATH:
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        # save_model_to_json(model, f'./model/{THIS_FILE_NAME}.json')

    # ===
    # 运行时数据保存
    # ===

    fig = plt.figure()
    plot_train,  = plt.plot(history['train'])
    plot_val, = plt.plot(history['val'])
    plt.legend(handles=[plot_train, plot_val], labels=['train', 'val'], loc='best')
    plt.show()
    fig.savefig(f'./runtime/loss_epochs/{THIS_FILE_NAME}.png')

# ===
# 其他
# ===

print("运行完毕")

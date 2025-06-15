import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import CNNPATHMNIST
import config  # 新增

# 1. 加载PATHMNIST数据
npz = np.load('data/pathmnist.npz')
x_train = npz['train_images']  # (N, 28, 28, 3)
y_train = npz['train_labels'].flatten()  # (N,)
x_test = npz['test_images']
y_test = npz['test_labels'].flatten()

# 转为torch tensor，调整通道顺序
x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float() / 255.0  # (N, 3, 28, 28)
x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255.0
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)
train_loader = DataLoader(trainset, batch_size=config.LOCAL_BS, shuffle=True)  # 用config里的batch
test_loader = DataLoader(testset, batch_size=config.TEST_BATCH_SIZE, shuffle=False)

# 2. 初始化模型、损失、优化器
device = config.DEVICE
model = CNNPATHMNIST().to(device)
criterion = config.CRITERION
optimizer = optim.SGD(model.parameters(), lr=config.LOCAL_LR, momentum=config.LOCAL_MOMENTUM, weight_decay=5e-4)

# 3. 训练与测试函数
def test(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return correct / total, loss_sum / total

# 4. 训练主循环
for epoch in range(1, config.LOCAL_EPOCHS+1):  # 用config里的本地epoch
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc, loss = test(model, test_loader)
    print(f'Epoch {epoch}: Test Acc={acc*100:.2f}%, Test Loss={loss:.4f}')
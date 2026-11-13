import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== UNet 模型 ==========
class UNet(nn.Module):
    def __init__(self, in_channels=13, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        mid = self.middle(self.pool4(d4))

        u4 = self.up4(mid)
        u4 = self.dec4(torch.cat([u4, d4], dim=1))

        u3 = self.up3(u4)
        u3 = self.dec3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1))

        out = self.final(u1)
        return torch.sigmoid(out)  # 输出概率

# ========== 自定义 Dataset ==========
class FAWDataset(Dataset):
    def __init__(self, data_folder, label_folder):
        self.data_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.nc')])
        self.label_folder = label_folder

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 读取 .nc 文件
        nc_path = self.data_files[idx]
        ds = xr.open_dataset(nc_path)
        data = ds.to_array().values  # shape: (13, 360, 720)
        ds.close()

        # 读取标签掩膜，假设为 npz 格式，shape: (360, 720)
        base_name = os.path.basename(nc_path).replace('.nc', '.npz')
        label_path = os.path.join(self.label_folder, base_name)
        label = np.load(label_path)['arr_0']  # shape: (360, 720)

        # 扩展为 [1, 360, 720]
        label = np.expand_dims(label, axis=0)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ========== 训练与验证函数 ==========
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == y).sum().item()
            total += torch.numel(y)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# ========== 主函数 ==========
def main():
    # 路径设置
    train_data_dir = r"E:\FAW\data\train"
    val_data_dir = r"E:\FAW\data\val"
    label_dir = r"E:\FAW\labels"

    # 超参数
    batch_size = 2
    num_epochs = 50
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_set = FAWDataset(train_data_dir, label_dir)
    val_set = FAWDataset(val_data_dir, label_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 模型、损失、优化器
    model = UNet(in_channels=13, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "unet_faw_model.pth")

if __name__ == "__main__":
    main()

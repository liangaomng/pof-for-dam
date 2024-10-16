import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
#保存模型

def save_model(model, optimizer, epoch, save_path='model.pth'):
    """
    保存模型的权重和优化器的状态。
    
    Args:
    - model (torch.nn.Module): 需要保存的模型
    - optimizer (torch.optim.Optimizer): 需要保存的优化器
    - epoch (int): 当前 epoch
    - save_path (str): 模型保存的文件路径
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, save_path)
    print(f"Model saved at {save_path}")

def load_model(model, load_path='model.pth', device="cuda:0"):
    """
    加载模型的权重和优化器的状态。
    
    Args:
    - model (torch.nn.Module): 需要加载权重的模型
    - optimizer (torch.optim.Optimizer): 需要加载状态的优化器
    - load_path (str): 保存模型的文件路径
    - device (str): 加载模型到的设备 ('cpu' 或 'cuda')
    
    Returns:
    - model: 加载了权重的模型
    - optimizer: 加载了状态的优化器
    - epoch: 加载的epoch，方便继续训练
    """
    # 加载保存的模型字典
    checkpoint = torch.load(load_path, map_location=device)
    
    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    

    # 恢复训练的epoch
    epoch = checkpoint['epoch']
    
    print(f"Model loaded from {load_path}, starting from epoch {epoch}")
    
    # 返回加载后的模型，优化器，以及从哪个epoch继续训练
    return model
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select the sample and convert to tensor
        sample = self.data[idx]
        input = sample[0:1]  # Select the first time step as input
        output = sample[1:]  # Select the next 30 time steps as output
        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)
def save_fig(tensor):
   steps = [0, 5, 10,20,25, 29]  # 因为Python的索引从0开始，所以实际上是第1步，第10步，第20步，第30步
   fig, axes = plt.subplots(1, len(steps), figsize=(8, 6))  # 创建一个1行多列的图形，每列放一个步骤的图像
   fig.suptitle('Outputs of Steps',  fontsize=16, y=0.85)
   # 在子图中显示每个步骤的图像
   for ax, step in zip(axes, steps):
      im = ax.imshow(tensor[0, step, :, :].cpu().detach().numpy(), extent=(0, 0.61, 0, 1.6), origin='lower', cmap='jet')
      ax.axis('off')  # 关闭坐标轴显示
      ax.set_title(f"Step {step+1}")  # 添加标题，显示每个子图对应的时间步，因为步数从0开始，显示时加

   # 调整子图之间的间距
   plt.subplots_adjust(wspace=0.01, hspace=0)
   plt.tight_layout(pad=0.4)
   # 在最后一个子图旁边添加颜色条
   # 添加颜色条，设置aspect和shrink调整颜色条的大小和形状
   cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.07, aspect=30, shrink=0.8)
   cbar.set_label('Depth')


   # 保存图像，包含紧凑边框调整
   plt.savefig("output.png", dpi=300, bbox_inches='tight')


class SpatialTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=4, dim_feedforward=256, dropout=0.05):
        super().__init__()
        
        # Transformer encoder layer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Linear layers for encoding and decoding spatial data
        self.encoder = nn.Linear(128 * 128, d_model)  # Encoding the spatial data
        self.decoder = nn.Linear(d_model, 128 * 128)  # Decoding back to spatial data

        # Layer normalization for stabilization
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: Encode the input's spatial dimensions into the model's feature space
        x = x.flatten(start_dim=2)  # Flatten spatial dimensions [batch, 1, 128*128]
        x = self.encoder(x)  # [batch, 1, d_model]
        x = self.encoder_norm(x)  # Apply layer normalization after encoding

        # Step 2: Create a placeholder for the sequence prediction (input step + future steps)
        seq_len = 30  # We predict 30 steps into the future
        predictions = torch.zeros(batch_size, seq_len, x.size(2), device=x.device)  # Placeholder for predictions

        # Step 3: Use the first time step as the initial input
        predictions[:, 0, :] = x[:, 0, :]  # Initial input to start the sequence

        # Step 4: Generate future steps iteratively (autoregressive prediction)
        for t in range(1, seq_len):
            # Pass through the Transformer encoder with layer normalization
            x = predictions[:, :t, :].clone()  # Select previous steps
            x = self.transformer_encoder(x)
            x = self.decoder_norm(x)  # Normalize after Transformer encoder

            # Predict the next step based on the previous steps
            next_step = x[:, -1, :]  # Take the last step's output from the transformer

            # Add the predicted next step to the predictions
            predictions[:, t, :] = next_step

        # Step 5: Decode the predictions back into the original spatial dimensions
        output = self.decoder(predictions)  # [batch, 30, d_model]
        output = output.view(batch_size, seq_len, 128, 128)  # Reshape to [batch, 30, 128, 128]

        return output
# 自定义相对二范数损失函数
def relative_l2_loss(output, target):
    """
    计算相对二范数损失函数
    Args:
        output: 模型的预测输出 [batch, ...]
        target: 真实值目标 [batch, ...]
    Returns:
        loss: 相对二范数损失
    """
    # 计算二范数
    numerator = torch.norm(output - target, p=2)  # ||output - target||_2
    denominator = torch.norm(target, p=2)  # ||target||_2
    
    # 防止除以0的情况
    if denominator == 0:
        return numerator  # 如果目标全为0，返回标准的L2损失
    else:
        return numerator / denominator  # 计算相对二范数
if __name__ == "__main__":
    # 检查 CUDA 是否可用
    # Load and reshape data
    data = torch.load("/root/NC_Dataset/2d/dam2d/tri_mask.pth")[:, :, 0, :, :]
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=40, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss, and optimizer
    model = SpatialTransformer().to(device)  # 将模型转移到 GPU
    criterion = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Initialize scheduler: 每隔30个epoch, 学习率缩小0.95倍
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=100, gamma=0.95)
    # Training loop


    #加载模型和优化器状态
    model = load_model(model, load_path='tri_model_epoch_999.pth', device=device)
    for epoch in range(1000):
    
        for input, target in dataloader:
                # 将输入和目标转移到 GPU
                input = input.to(device)
                target = target.to(device)

                # Forward pass through the model
                output = model(input)  # [batch, 30, 128, 128]

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss = 1*relative_l2_loss(output, target)
                loss.backward()
                optimizer.step()
            # 每个 epoch 后更新学习率
        scheduler.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # epoch 后保存模型
    save_model(model, optimizer, epoch, save_path=f'tri_model_epoch_{epoch}.pth')
    save_fig(output)
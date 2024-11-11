import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 设置参数
time_steps = 1000  # 时间步数
features = 64      # 特征数
num_classes = 10   # 类别数量

# 超参数（需要调整的数值）
learning_rate = 0.001  # 学习率
num_epochs = 20        # 训练轮数
batch_size = 32        # 批次大小z

# 自定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__() 
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_length, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # 对时间步进行softmax
        weighted = x * attn_weights  # 加权输入
        output = torch.sum(weighted, dim=1)  # 对序列维度求和，得到整体表示
        return output

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # BiGRU层
        self.bigru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        
        # 注意力机制
        self.attention = Attention(input_dim=256)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x形状: (batch_size, time_steps, features)
        # 调整维度以匹配Conv1d的输入格式
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, features, time_steps)

        # 卷积和池化层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))

        # 调整维度以匹配GRU的输入格式
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length, features)

        # BiGRU层
        gru_out, _ = self.bigru(x)  # gru_out形状: (batch_size, seq_length, hidden_size * 2)

        # 注意力机制
        attn_out = self.attention(gru_out)  # (batch_size, hidden_size * 2)

        # 全连接层
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        x = self.fc2(x)
        # 不需要在这里应用softmax，CrossEntropyLoss会处理
        return x

# 创建模型实例
model = MyModel()

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

# =====================================
# 模拟输入数据和标签（在主代码中进行输入）
# =====================================

# 生成模拟数据（需要替换为实际数据）
# 输入数据: (样本数量, 时间步数, 特征数)
num_samples = 1000  # 样本数量（需要根据实际情况调整）

# 生成随机输入和标签
inputs = torch.randn(num_samples, time_steps, features)
labels = torch.randint(0, num_classes, (num_samples,))

# 创建数据集和数据加载器
dataset = TensorDataset(inputs, labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 此处batch_size已标注为批次大小

# =====================================
# 训练过程
# =====================================

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累积损失
    running_corrects = 0  # 累积正确预测数

    for batch_inputs, batch_labels in data_loader:
        # 将数据移动到设备
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # 前向传播
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        # 统计损失和准确率
        running_loss += loss.item() * batch_inputs.size(0)
        _, preds = torch.max(outputs, 1)  # 获取预测结果
        running_corrects += torch.sum(preds == batch_labels.data)

    # 计算平均损失和准确率
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects.double() / num_samples

    print(f'第 {epoch+1} 个周期，Loss: {epoch_loss:.4f}，Accuracy: {epoch_acc:.4f}')

print('训练完成。')
print('模型结构：')
print(model)
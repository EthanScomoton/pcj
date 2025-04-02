# 港口综合能源系统优化

本项目实现了一个港口综合能源系统的优化分析工具，包括可再生能源管理、电网调度和储能系统优化。

## 项目结构

- `Full_Model/` - 整体系统模型
  - `main.py` - 主程序入口
  - `IES.py` - 综合能源系统类
  - `BES.py` - 电池储能系统类
  - `OSS.py` - 储能规模优化
  - `EO.py` - 能源优化器
  - `REO.py` - 可再生能源优化器
  - `EF.py` - 特征提取工具
  - `All_Models_EGrid_Paper.py` - 预测模型定义
  - `train_model.py` - 模型训练脚本
  - `convert_model.py` - 模型转换脚本（特征维度调整）

- `All_Models/` - 各种能源预测模型

## 安装依赖

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

## 使用方法

### 运行完整分析

```bash
cd Full_Model
python main.py
```

### 解决模型特征维度不匹配问题

项目中的预训练模型与当前数据集的特征维度可能不匹配，提供了两种解决方案：

#### 方案1: 使用模型转换脚本

这个方法将已有的预训练模型调整为匹配当前数据集特征维度的新模型。

```bash
cd Full_Model
python convert_model.py
```

运行后将生成一个名为`current_EModel_FeatureWeight4.pth`的兼容模型。

#### 方案2: 使用当前数据重新训练模型

这个方法使用当前数据集直接训练一个新模型。

```bash
cd Full_Model
python train_model.py
```

训练完成后将生成一个名为`best_EModel_FeatureWeight4.pth`的新模型。

#### 方案3: 使用特征适配

已在`IES.py`中实现了特征适配函数，可以在运行时自动调整特征维度以匹配模型要求。

## 主要功能

- **电网负荷预测**：使用深度学习模型预测未来电网负荷
- **储能规模优化**：基于经济性和技术性指标选择最优储能规模
- **能源调度**：实时优化储能系统充放电策略
- **可再生能源管理**：优化太阳能和风能的使用
- **系统性能评估**：计算关键绩效指标并可视化结果

## 错误排查

### 模型维度不匹配错误

如果遇到类似以下错误：

```
size mismatch for feature_importance: copying a param with shape torch.Size([22]) from checkpoint, the shape in current model is torch.Size([26]).
```

这表示预训练模型的特征维度（22）与当前模型使用的特征维度（26）不匹配。可以使用上述提供的任何解决方案修复此问题。

## 系统架构

港口综合能源系统由以下组件组成：

1. **预测模块**：预测未来能源需求和可再生能源生成
2. **能源优化模块**：优化系统整体能源流
3. **储能管理模块**：控制储能系统的充放电
4. **可再生能源管理**：优化可再生能源的使用
5. **分析与可视化**：评估系统性能并可视化结果

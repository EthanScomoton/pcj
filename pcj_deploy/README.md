# 能源预测系统部署

这是一个基于深度学习的能源预测系统，使用LSTM和注意力机制进行电网能源补偿值的预测。

## 项目结构

```
pcj_deploy/
├── app.py                  # Flask Web应用主文件
├── create_dummy_model.py   # 创建示例模型的脚本
├── models/                 # 模型相关文件
│   ├── __init__.py         # 包初始化文件
│   ├── energy_models.py    # 模型定义
│   ├── data_processor.py   # 数据处理类
│   ├── predictor.py        # 预测器类
│   └── best_model.pth      # 模型权重文件（运行后生成）
├── static/                 # 静态文件目录
│   └── sample_data.csv     # 示例数据（运行后生成）
└── templates/              # HTML模板
    └── index.html          # Web界面
```

## 安装依赖

```bash
pip install torch numpy pandas flask scikit-learn
```

## 使用方法

1. 创建示例模型（如果没有实际模型）：

```bash
cd pcj_deploy
python create_dummy_model.py
```

2. 启动Web应用：

```bash
python app.py
```

3. 在浏览器中访问：`http://localhost:5000`

## Web界面功能

- **模型管理**：上传和加载模型文件
- **数据处理器配置**：上传训练数据以拟合数据处理器
- **预测功能**：上传数据进行预测，或生成示例数据
- **结果可视化**：以表格和图表形式展示预测结果

## 数据格式要求

上传的CSV文件应包含以下列（至少部分列）：

- timestamp：时间戳
- season：季节
- holiday：是否假日
- weather：天气
- temperature：温度
- working_hours：是否工作时间
- E_wind：风能
- E_PV：光伏能源
- v_wind：风速
- wind_direction：风向
- E_grid：电网能源补偿值（目标变量）

## 部署到远程服务器

1. 将整个`pcj_deploy`目录复制到远程服务器

2. 安装依赖：
```bash
pip install torch numpy pandas flask scikit-learn
```

3. 启动应用：
```bash
cd pcj_deploy
python app.py
```

4. 配置防火墙以允许5000端口访问（如果需要）

5. 在生产环境中，建议使用Gunicorn或uWSGI作为WSGI服务器，并配合Nginx作为反向代理
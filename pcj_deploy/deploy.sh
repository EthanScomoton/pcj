#!/bin/bash

# 能源预测系统部署脚本

echo "===== 能源预测系统部署脚本 ====="

# 检查Python环境
echo "检查Python环境..."
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "错误: 未找到Python。请安装Python 3.6或更高版本。"
    exit 1
fi

echo "使用Python: $($PYTHON --version)"

# 创建虚拟环境
echo "创建虚拟环境..."
$PYTHON -m venv venv
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install --upgrade pip
pip install torch numpy pandas flask scikit-learn

# 创建示例模型
echo "创建示例模型..."
$PYTHON create_dummy_model.py

# 启动应用
echo "启动应用..."
$PYTHON app.py
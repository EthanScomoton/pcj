# 使用Python 3.9作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 安装依赖
RUN pip install --no-cache-dir torch numpy pandas flask scikit-learn

# 创建示例模型
RUN python create_dummy_model.py

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"]
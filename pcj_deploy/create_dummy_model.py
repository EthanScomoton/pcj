import torch
import os
from models.energy_models import EModel_FeatureWeight5

def create_dummy_model():
    """
    创建一个示例模型并保存到models目录
    """
    print("正在创建示例模型...")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建模型实例
    feature_dim = 18  # 假设特征维度为18
    model = EModel_FeatureWeight5(
        feature_dim=feature_dim,
        window_size=20,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.2
    )
    
    # 保存模型
    model_path = os.path.join('models', 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"示例模型已保存到: {model_path}")

if __name__ == "__main__":
    create_dummy_model()
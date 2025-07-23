import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from models.predictor import EnergyPredictor

app = Flask(__name__)

# 初始化预测器
model_path = os.path.join('models', 'best_model.pth')
predictor = None

# 如果模型文件存在，则加载模型
if os.path.exists(model_path):
    try:
        predictor = EnergyPredictor(model_path=model_path)
    except Exception as e:
        print(f"加载模型时出错: {e}")

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    if predictor is None:
        return jsonify({'error': '模型未加载，请先上传模型文件'}), 400
    
    try:
        # 获取上传的CSV文件
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 进行预测
        predictions = predictor.predict(df)
        
        # 返回预测结果
        result = {
            'predictions': predictions.tolist(),
            'message': '预测成功'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'预测过程中出错: {str(e)}'}), 500

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """上传模型文件"""
    try:
        # 获取上传的模型文件
        if 'model_file' not in request.files:
            return jsonify({'error': '没有上传模型文件'}), 400
        
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({'error': '未选择模型文件'}), 400
        
        # 保存模型文件
        model_file.save(model_path)
        
        # 重新加载预测器
        global predictor
        predictor = EnergyPredictor(model_path=model_path)
        
        return jsonify({'message': '模型上传并加载成功'})
    
    except Exception as e:
        return jsonify({'error': f'上传模型时出错: {str(e)}'}), 500

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """上传训练数据以拟合数据处理器"""
    if predictor is None:
        return jsonify({'error': '模型未加载，请先上传模型文件'}), 400
    
    try:
        # 获取上传的CSV文件
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 拟合数据处理器
        predictor.fit_processor(df)
        
        return jsonify({'message': '数据处理器拟合成功'})
    
    except Exception as e:
        return jsonify({'error': f'拟合数据处理器时出错: {str(e)}'}), 500

@app.route('/generate_sample_data', methods=['GET'])
def generate_sample_data():
    """生成示例数据"""
    try:
        # 创建示例数据
        n_samples = 100
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        data = {
            'timestamp': timestamps,
            'season': np.random.choice(['spring', 'summer', 'autumn', 'winter'], size=n_samples),
            'holiday': np.random.choice(['yes', 'no'], size=n_samples),
            'weather': np.random.choice(['sunny', 'cloudy', 'rainy', 'snowy'], size=n_samples),
            'temperature': np.random.uniform(0, 30, size=n_samples),
            'working_hours': np.random.choice(['yes', 'no'], size=n_samples),
            'E_wind': np.random.uniform(0, 100, size=n_samples),
            'E_PV': np.random.uniform(0, 100, size=n_samples),
            'v_wind': np.random.uniform(0, 20, size=n_samples),
            'wind_direction': np.random.uniform(0, 360, size=n_samples),
            'E_grid': np.random.uniform(1000, 5000, size=n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # 保存为CSV文件
        sample_path = os.path.join('static', 'sample_data.csv')
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df.to_csv(sample_path, index=False)
        
        return jsonify({'message': '示例数据生成成功', 'path': '/static/sample_data.csv'})
    
    except Exception as e:
        return jsonify({'error': f'生成示例数据时出错: {str(e)}'}), 500

if __name__ == '__main__':
    # 确保static和templates目录存在
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)
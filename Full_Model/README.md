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

# Port Integrated Energy System Optimization Project

This project implements an optimization analysis tool for a Port Integrated Energy System (PIES), including renewable energy management, grid scheduling, and energy storage system (BESS) optimization.

## Project Goal
The goal is to optimize the energy management of a port by leveraging renewable energy sources (PV, Wind) and a Battery Energy Storage System (BESS) to reduce grid dependency, peak loads, and operational costs. The system uses deep learning models to predict energy demand and mathematical optimization (Convex Optimization) to schedule battery operations.

## Key Features
- **Load Prediction**: Uses LSTM/GRU/Transformer-based models to predict future port energy demand.
- **Storage Sizing Optimization**: Determines the optimal capacity (kWh) and power (kW) for the BESS based on economic indicators (NPV, IRR, Payback Period).
- **Operational Scheduling**: Real-time optimization of battery charging/discharging strategies to minimize electricity costs (Arbitrage) and peak demand.
- **Renewable Integration**: Maximizes the utilization of local solar and wind energy.
- **Simulation & Analysis**: Simulates system operation over time and calculates Key Performance Indicators (KPIs).

## Project Structure
- `Full_Model/`: Core system implementation.
  - `main.py`: Main entry point for running the simulation and optimization.
  - `IES.py`: `IntegratedEnergySystem` class, managing the simulation loop and prediction integration.
  - `BES.py`: `BatteryEnergyStorage` class, modeling battery physics (SOC, efficiency).
  - `EO.py`: `EnergyOptimizer`, implementing the scheduling algorithm.
  - `OSS.py`: Storage size optimization logic.
  - `EF.py`: Feature extraction and utility functions.
  - `All_Models_EGrid_Paper.py`: Definition of deep learning prediction models.
- `All_Models/`: Contains various experimental prediction models.

## Usage

### Prerequisites
Ensure you have the following Python packages installed:
```bash
pip install torch pandas numpy matplotlib scikit-learn cvxpy
```

### Running the Optimization
To run the complete analysis, including storage sizing and operational simulation:

```bash
cd Full_Model
python main.py
```

This will:
1. Load and preprocess the data.
2. Initialize and load the pre-trained demand prediction model.
3. **Crucial Step**: Fit data scalers to ensure model inputs/outputs are correctly normalized (fixing previous issues with data scaling).
4. Run an optimization loop to find the best BESS size (Capacity & Power).
5. Simulate the optimal system operation over a period.
6. Compare with a baseline (no storage) system.
7. Display KPIs and visualizations.

### Troubleshooting "Predicted Data Not Participating"
If you encounter issues where the optimization seems to ignore predictions (e.g., flat 0 results):
- This has been addressed by ensuring `feature_cols` are explicitly passed to all modules to prevent feature scrambling.
- Data scalers (`StandardScaler`) are now properly fitted in `main.py` and passed to the simulation engine to handle the transformation between raw values (kW) and model inputs (normalized).

## Methodology
1. **Prediction**: The system predicts the next 24h load.
2. **Optimization**:
   - Objective: Minimize Grid Cost + Penalties.
   - Constraints: Power balance, Battery SOC limits, Charge/Discharge limits.
   - Solver: CVXPY (using OSQP or default solver).

## Recent Updates
- Fixed feature alignment issue where model inputs were scrambled.
- Implemented correct data scaling/descaling flow for the prediction model.
- Enhanced `IES` and `OSS` modules to support explicit feature column definitions.


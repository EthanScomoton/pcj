import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# 定义状态点
states = {
    '1': {'T': 273.15 + 5, 'Q': 1},  # 蒸发器出口，气态
    '2': {'P': 9e6},                 # 压缩机出口，超临界
    '3': {'P': 9e6, 'T': 273.15 + 35},  # 气体冷却器出口，超临界
    '4': {'P': 4e6}                  # 节流阀出口
}

# 计算状态点的其他参数
for key, state in states.items():
    try:
        if 'Q' in state:  # 如果定义了Q，则计算P, H, S, D
            state['P'] = PropsSI('P', 'T', state['T'], 'Q', state['Q'], 'CO2')
            state['H'] = PropsSI('H', 'T', state['T'], 'Q', state['Q'], 'CO2')
            state['S'] = PropsSI('S', 'T', state['T'], 'Q', state['Q'], 'CO2')
            state['D'] = PropsSI('D', 'T', state['T'], 'Q', state['Q'], 'CO2')
        elif 'P' in state and 'T' in state:  # 如果定义了P和T
            state['H'] = PropsSI('H', 'P', state['P'], 'T', state['T'], 'CO2')
            state['S'] = PropsSI('S', 'P', state['P'], 'T', state['T'], 'CO2')
            state['D'] = PropsSI('D', 'P', state['P'], 'T', state['T'], 'CO2')
        elif 'P' in state and 'S' in state:  # 如果定义了P和S
            state['H'] = PropsSI('H', 'P', state['P'], 'S', state['S'], 'CO2')
            state['T'] = PropsSI('T', 'P', state['P'], 'S', state['S'], 'CO2')
            state['D'] = PropsSI('D', 'P', state['P'], 'S', state['S'], 'CO2')
        elif 'P' in state and 'H' in state:  # 如果定义了P和H
            state['T'] = PropsSI('T', 'P', state['P'], 'H', state['H'], 'CO2')
            state['S'] = PropsSI('S', 'P', state['P'], 'H', state['H'], 'CO2')
            state['D'] = PropsSI('D', 'P', state['P'], 'H', state['H'], 'CO2')
    except Exception as e:
        print(f"Error calculating properties for state {key}: {e}")
    finally:
        # 确保所有状态点的参数被正确初始化
        if 'S' not in state:
            state['S'] = None
        if 'T' not in state:
            state['T'] = None
        if 'H' not in state:
            state['H'] = None
        if 'D' not in state:
            state['D'] = None

# 检查所有状态点是否初始化成功
for key, state in states.items():
    print(f"State {key}: {state}")

# 绘制T-S图
plt.figure()
try:
    for i in range(1, 5):
        if states[str(i)]['S'] is not None and states[str((i % 4) + 1)]['S'] is not None:
            plt.plot(
                [states[str(i)]['S'], states[str((i % 4) + 1)]['S']],
                [states[str(i)]['T'] - 273.15, states[str((i % 4) + 1)]['T'] - 273.15],
                'bo-'
            )
    plt.xlabel('熵 (J/kg·K)')
    plt.ylabel('温度 (°C)')
    plt.title('CO₂热泵系统的T-S图')
    plt.grid()
    plt.show()
except Exception as e:
    print(f"Error plotting T-S diagram: {e}")

# 绘制P-V图
plt.figure()
try:
    for i in range(1, 5):
        if states[str(i)]['D'] is not None and states[str((i % 4) + 1)]['D'] is not None:
            plt.plot(
                [1 / states[str(i)]['D'], 1 / states[str((i % 4) + 1)]['D']],
                [states[str(i)]['P'] / 1e6, states[str((i % 4) + 1)]['P'] / 1e6],
                'bo-'
            )
    plt.xlabel('比容 (m³/kg)')
    plt.ylabel('压力 (MPa)')
    plt.title('CO₂热泵系统的P-V图')
    plt.grid()
    plt.show()
except Exception as e:
    print(f"Error plotting P-V diagram: {e}")
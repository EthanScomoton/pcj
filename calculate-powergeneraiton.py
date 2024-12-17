import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import pandas as pd
from datetime import datetime

## main
latitude = None
day_of_year = None
edge_indices = None
energy_matrix = None
start_point = 1

load_data_path = "/Users/ethan/Desktop/load_data.csv"
load_data_df = pd.read_csv(load_data_path)


def calculate_total_energy(input_row):

    material_type = input_row["dock_position"]
    Q_total = input_row["ship_grade"] 
    end_point = input_row["destination"]

    latitude = 30  # 设置纬度
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
    hour_angle = np.degrees(np.arccos(-np.tan(np.deg2rad(latitude)) * np.tan(np.deg2rad(declination))))
    sunrise = 12 - hour_angle / 15
    sunset = 12 + hour_angle / 15
    sunrise = max(0, sunrise)
    sunset = min(24, sunset)
    print(f'当天日出时间：{sunrise:.2f} 点')
    print(f'当天日落时间：{sunset:.2f} 点')

    P_solar_one = 0.4  # 单块最大功率（kW）
    P_solar_Panel = 200
    P_solar_max = P_solar_one * P_solar_Panel
    P_solar = np.zeros_like(t)

    for i in range(len(t)):
        current_time = t[i] % 24
        if sunrise <= current_time <= sunset:
            P_solar[i] = P_solar_max * np.sin(np.pi * (current_time - sunrise) / (sunset - sunrise))
    P_solar[P_solar < 0] = 0
    E_solar = P_solar * dt

    k_weibull = 2  # 形状参数
    c_weibull = 8  # 平均风速
    np.random.seed(1)
    v_wind = weibull_min.rvs(k_weibull, scale=c_weibull, size=len(t))
    v_in = 5
    v_rated = 8
    v_out = 12
    P_wind_rated = 1000
    N_wind_turbine = 3
    P_wind = np.zeros_like(t)

    for i in range(len(v_wind)):
        v = v_wind[i]
        if v < v_in or v >= v_out:
            P_wind[i] = 0
        elif v_in <= v < v_rated:
            P_wind[i] = P_wind_rated * ((v - v_in) / (v_rated - v_in)) ** 3
        else:
            P_wind[i] = P_wind_rated

    P_wind *= N_wind_turbine
    E_wind = P_wind * dt
    E_renewable = E_solar + E_wind


    E_max = 50000  # 储能系统最大容量（kWh）
    E_storage = np.zeros_like(t)
    E_storage[0] = E_max  # 初始储能水平
    P_charge_max = 1000
    P_discharge_max = P_charge_max
    E_charge_max = P_charge_max * dt
    E_discharge_max = P_discharge_max * dt
    E_target = E_max * 0.8


    E_solar_supply = np.zeros_like(t)
    E_wind_supply = np.zeros_like(t)
    E_storage_discharge = np.zeros_like(t)
    E_grid_supply = np.zeros_like(t)
    E_storage_charge_from_renewable = np.zeros_like(t)
    E_storage_charge_from_grid = np.zeros_like(t)
    E_grid_draw = np.zeros_like(t)

    for i in range(1, len(t)):
        E_gen_solar = E_solar[i]
        E_gen_wind = E_wind[i]
        E_load_step = E_load[i] * dt
        E_storage[i] = E_storage[i-1]
        E_solar_supply[i] = min(E_gen_solar, E_load_step)
        remaining_load = E_load_step - E_solar_supply[i]
        E_wind_supply[i] = min(E_gen_wind, remaining_load)
        remaining_load -= E_wind_supply[i]
        net_renewable_energy = (E_gen_solar + E_gen_wind) - (E_solar_supply[i] + E_wind_supply[i])

        if net_renewable_energy > 0:
            E_storage_charge_from_renewable[i] = min([net_renewable_energy, E_charge_max, E_max - E_storage[i]])
            E_storage[i] += E_storage_charge_from_renewable[i]
            E_storage_charge_from_grid[i] = 0
            E_grid_supply[i] = 0
            E_storage_discharge[i] = 0
        else:
            deficit_load = remaining_load
            E_storage_discharge[i] = min([deficit_load, E_discharge_max, E_storage[i]])
            E_storage[i] -= E_storage_discharge[i]
            deficit_load -= E_storage_discharge[i]

            if deficit_load > 0:
                E_grid_supply[i] = deficit_load
                E_grid_draw[i] += E_grid_supply[i]

            if E_storage[i] < E_target:
                E_storage_charge_from_grid[i] = min([E_charge_max, E_target - E_storage[i]])
                E_storage[i] += E_storage_charge_from_grid[i]
                E_grid_draw[i] += E_storage_charge_from_grid[i]
            else:
                E_storage_charge_from_grid[i] = 0

            E_storage_charge_from_renewable[i] = 0

        E_storage[i] = min(max(E_storage[i], 0), E_max)

    return E_solar, E_wind, E_renewable, 



# 使用 load_data_df 的数据逐行计算总能耗
load_data_df["target"] = load_data_df.apply(calculate_total_energy, axis=1)
pd.set_option('display.max_rows', None)        # 显示所有行
pd.set_option('display.max_columns', None)     # 显示所有列
pd.set_option('display.width', None)           # 根据窗口自动调整宽度
pd.set_option('display.max_colwidth', None)    # 不限制列宽
print(load_data_df)
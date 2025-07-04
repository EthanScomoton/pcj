import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import pandas as pd
from datetime import datetime

## 全局变量（如有需要，可根据项目结构改为函数传参）
latitude = None
day_of_year = None
edge_indices = None
energy_matrix = None
start_point = 1

load_data_path = "/Users/ethanzhu/Desktop/load_data10.csv"
load_data_df = pd.read_csv(load_data_path)

renewable_data_path = "/Users/ethanzhu/Desktop/renewable_data10.csv"
renewable_data_df = pd.read_csv(renewable_data_path)

def calculate_total_energy(input_row):
    """
    根据输入行（含时间戳、起始点、终止点、货物种类等），计算本次的总能耗及可再生能源供电情况。
    当从 renewable_data_df 中检测到:
        1) holiday == 'yes' -> 太阳能、风能全部输出为 0
        2) weather == 'cloudy' -> 仅太阳能输出为 0
        3) weather == 'windy'  -> 仅风能输出为 0
        4) weather in ('rainy', 'snowy') -> 太阳能、风能全部输出为 0
    其余逻辑保持不变。
    """

    material_type = input_row["dock_position"]
    Q_total = input_row["ship_grade"]
    end_point = input_row["destination"]
    day_of_yearst = input_row["timestamp"]
    timestamp = datetime.strptime(day_of_yearst, "%m/%d/%Y %H:%M")
    day_of_year = timestamp.timetuple().tm_yday

    # ===== 在这里尝试获取本行对应的可再生能源信息（holiday & weather） =====
    row_renewable = renewable_data_df.loc[renewable_data_df['timestamp'] == day_of_yearst]
    if not row_renewable.empty:
        holiday_value = row_renewable['holiday'].values[0]
        weather_value = row_renewable['weather'].values[0]
    else:
        holiday_value = 'no'
        weather_value = 'normal'
    # ====================================================================

    def dijkstra(energy_matrix, start_point, end_point):
        num_nodes = energy_matrix.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        distance = np.full(num_nodes, np.inf)
        previous = np.full(num_nodes, np.nan)
        distance[start_point] = 0

        pq = [(0, start_point)]

        while pq:
            current_dist, current = heapq.heappop(pq)
            if visited[current]:
                continue
            visited[current] = True

            if current == end_point:
                break

            for neighbor in range(num_nodes):
                if (not visited[neighbor]) and (energy_matrix[current, neighbor] < np.inf):
                    new_distance = current_dist + energy_matrix[current, neighbor]
                    if new_distance < distance[neighbor]:
                        distance[neighbor] = new_distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_distance, neighbor))

        min_path = []
        c = end_point
        while not np.isnan(c):
            min_path.insert(0, int(c) + 1)
            c = previous[int(c)]

        E_total = distance[end_point]
        if np.isinf(E_total):
            min_path = []
        return min_path, E_total

    def calculate_energy_matrix(yard_matrix, Q_list_regular, P_list_regular, Q_total,
                                special_connections, material_type):
        n = yard_matrix.shape[0]
        energy_matrix = np.full((n, n), np.inf)
        edge_indices = []
        regular_idx = 0
        special_conn = special_connections.get(material_type, [])
        special_edge_dict = {
            (sc['nodes'][0] - 1, sc['nodes'][1] - 1): sc
            for sc in special_conn
        }

        for i in range(n):
            for j in range(i + 1, n):
                if yard_matrix[i, j] == 1:
                    special_key = (i, j) if (i, j) in special_edge_dict else (j, i)
                    if special_key in special_edge_dict:
                        sc = special_edge_dict[special_key]
                        Q_value = sc['Q']
                        P_value = sc['P']
                        if Q_value == 0:
                            continue
                        T_i = Q_total / Q_value
                        if P_value == 0:
                            P_value = 1e-6

                        if (np.isnan(P_value) or np.isnan(T_i)
                                or np.isinf(P_value) or np.isinf(T_i)):
                            print(f"无效的 P 或 T_i 值：P={P_value}, T_i={T_i}，发生在节点 {i+1} 到 {j+1}")
                            continue

                        energy_matrix[i, j] = energy_matrix[j, i] = P_value * T_i
                        edge_info = {'nodes': [i + 1, j + 1], 'Q': Q_value, 'P': P_value}
                        edge_indices.append(edge_info)
                    else:
                        if regular_idx >= len(Q_list_regular) or regular_idx >= len(P_list_regular):
                            raise ValueError('Q_list_regular 或 P_list_regular 长度不足，无法匹配所有常规连接。')

                        Q_value = Q_list_regular[regular_idx]
                        P_value = P_list_regular[regular_idx]
                        regular_idx += 1

                        if Q_value == 0 or np.isclose(Q_value, 0) or np.isinf(Q_value):
                            continue
                        T_i = Q_total / Q_value

                        if P_value == 0 or np.isclose(P_value, 0) or np.isinf(P_value):
                            continue

                        if (np.isnan(P_value) or np.isnan(T_i)
                                or np.isinf(P_value) or np.isinf(T_i)):
                            continue

                        energy_matrix[i, j] = energy_matrix[j, i] = P_value * T_i
                        edge_info = {'nodes': [i + 1, j + 1], 'Q': Q_value, 'P': P_value}
                        edge_indices.append(edge_info)

        return energy_matrix, edge_indices

    def calculate_energy_consumption(Q_total, material_type, start_point, end_point):
        Ei = None
        # ======= 以下是三种 yard 的邻接矩阵定义（保持原逻辑） =======
        mineral_yard_1_matrix = np.array([
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        ])
        mineral_yard_2_matrix = np.array([
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        ])
        coal_yard_matrix = np.array([
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0]
        ])

        Q_list_mineral_1 = np.concatenate([5000 * np.ones(7), 3500 * np.ones(2), 5000 * np.ones(14)])
        P_list_mineral_1 = np.concatenate([520 * np.ones(7), 90 * np.ones(2), 520 * np.ones(14)])
        Q_list_mineral_2 = np.concatenate([2500 * np.ones(15), [np.inf], 2500 * np.ones(3), [np.inf], 2500 * np.ones(3)])
        P_list_mineral_2 = np.concatenate([100 * np.ones(15), [np.inf], 100 * np.ones(3), [np.inf], 100 * np.ones(3)])
        Q_list_coal = np.concatenate([4500 * np.ones(26), [np.inf], 4100 * np.ones(5)])
        P_list_coal = np.concatenate([700 * np.ones(26), [np.inf], 644 * np.ones(5)])

        if material_type == 'mineral_1':
            yard_matrix = mineral_yard_1_matrix
            Q_list_regular = Q_list_mineral_1
            P_list_regular = P_list_mineral_1
        elif material_type == 'mineral_2':
            yard_matrix = mineral_yard_2_matrix
            Q_list_regular = Q_list_mineral_2
            P_list_regular = P_list_mineral_2
        elif material_type == 'coal':
            yard_matrix = coal_yard_matrix
            Q_list_regular = Q_list_coal
            P_list_regular = P_list_coal
        else:
            raise ValueError('未识别的堆放种类!')

        if start_point < 1 or start_point > yard_matrix.shape[0] or end_point < 1 or end_point > yard_matrix.shape[0]:
            raise ValueError('起始点或终止点超出矩阵范围！')

        special_connections = {
            'mineral_1': [{'nodes': [1, 18], 'Q': 5000, 'P': 280}],
            'mineral_2': [
                {'nodes': [1, 16], 'Q': 2500, 'P': 100},
                {'nodes': [1, 17], 'Q': 2500, 'P': 100},
                {'nodes': [2, 24], 'Q': 2500, 'P': 100},
                {'nodes': [10, 20], 'Q': 2500, 'P': 100},
                {'nodes': [9, 21], 'Q': 2500, 'P': 100}
            ],
            'coal': [
                {'nodes': [1, 20], 'Q': 4500, 'P': 700},
                {'nodes': [9, 26], 'Q': 4500, 'P': 700},
                {'nodes': [19, 32], 'Q': 4100, 'P': 644}
            ]
        }

        energy_matrix, edge_indices = calculate_energy_matrix(
            yard_matrix, Q_list_regular, P_list_regular,
            Q_total, special_connections, material_type
        )

        min_path, E_total = dijkstra(energy_matrix, start_point - 1, end_point - 1)

        # 计算 Ei（保持原逻辑）
        if material_type == 'mineral_1':
            Ei = E_total + Q_total / 5000 * 1065 + Q_total / 5000 * 355
        elif material_type == 'mineral_2':
            Ei = E_total + Q_total / 5000 * 1065 + Q_total / 5000 * 355
        elif material_type == 'coal':
            Ei = E_total + Q_total / 5000 * 1065 + Q_total / 5000 * 355 + Q_total / 4500 * 700

        return Ei, min_path, edge_indices

    # ====== 获取运输路径与消耗 ======
    Ei, min_path, edge_indices = calculate_energy_consumption(Q_total, material_type, start_point, end_point)

    # ====== 准备进行可再生能源模拟 ======
    start_time = 5
    dt = 0.05
    num_segments = len(min_path) - 1
    segment_info = [{'start_time': None, 'end_time': None, 'P_edge': None} for _ in range(num_segments)]
    Q_min_edge = np.inf
    total_P_edge = 0

    for k in range(num_segments):
        node_start = min_path[k]
        node_end = min_path[k + 1]
        edge_found = False
        for idx_e in range(len(edge_indices)):
            nodes = edge_indices[idx_e]['nodes']
            if ((nodes[0] == node_start and nodes[1] == node_end)
                    or (nodes[0] == node_end and nodes[1] == node_start)):
                Q_edge = edge_indices[idx_e]['Q']
                P_edge = edge_indices[idx_e]['P']
                edge_found = True
                break

        if not edge_found:
            raise ValueError(f'在 edge_indices 中未找到边 ({node_start}, {node_end})。可能原因：该边不可通行或数据有误。')

        Q_min_edge = min(Q_min_edge, Q_edge)
        total_P_edge += P_edge
        segment_info[k]['P_edge'] = P_edge

    T_total = Q_total / Q_min_edge
    T_end = T_total
    t = np.arange(start_time, start_time + T_end, dt)

    # ====== 负载功率（与时间步同长度）======
    E_load = np.full_like(t, total_P_edge)

    # ====== 太阳能计算（先按原逻辑得到 P_solar, E_solar）======
    latitude = 30
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
    hour_angle = np.degrees(np.arccos(-np.tan(np.deg2rad(latitude)) * np.tan(np.deg2rad(declination))))
    sunrise = 12 - hour_angle / 15
    sunset = 12 + hour_angle / 15
    sunrise = max(0, sunrise)
    sunset = min(24, sunset)

    P_solar_one = 0.4
    P_solar_Panel = 200
    P_solar_max = P_solar_one * P_solar_Panel
    P_solar = np.zeros_like(t)

    for i in range(len(t)):
        current_time = t[i] % 24
        if sunrise <= current_time <= sunset:
            P_solar[i] = P_solar_max * np.sin(np.pi * (current_time - sunrise) / (sunset - sunrise))
    P_solar[P_solar < 0] = 0
    E_solar = P_solar * dt

    # ====== 风能计算（先按原逻辑得到 P_wind, E_wind）======
    k_weibull = 2  # 形状参数
    c_weibull = 8  # 平均风速
    np.random.seed(1)
    v_wind = weibull_min.rvs(k_weibull, scale=c_weibull, size=len(t))
    # 添加风向信息，假设风向均匀分布在0到360度之间
    wind_direction = np.random.uniform(0, 360, size=len(t))
    optimal_wind_direction = 180  # 假设最佳风向为180度
 
    v_in = 5
    v_rated = 8
    v_out = 12
    P_wind_rated = 1000
    N_wind_turbine = 3
    P_wind = np.zeros_like(t)

    for i in range(len(v_wind)):
        v = v_wind[i]
        # 计算当前步长的风向影响因子
        current_dir = wind_direction[i]
        angle_diff = abs(current_dir - optimal_wind_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff  # 考虑360度周期性
        wind_factor = np.cos(np.deg2rad(angle_diff))
        wind_factor = max(wind_factor, 0)  # 避免余弦为负，风向不利时功率降为0

        if v < v_in or v >= v_out:
            power = 0
        elif v_in <= v < v_rated:
            power = P_wind_rated * ((v - v_in) / (v_rated - v_in)) ** 3
        else:
            power = P_wind_rated

        # 将计算出的风速功率乘以风向影响因子
        P_wind[i] = power * wind_factor

    P_wind *= N_wind_turbine
    E_wind = P_wind * dt

        # =============== 根据 holiday/weather 置零 ===============
    if holiday_value == 'yes':
        # 节假日 -> 所有可再生能源都置 0
        P_solar = np.zeros_like(P_solar)
        E_solar = np.zeros_like(E_solar)
        P_wind = np.zeros_like(P_wind)
        E_wind = np.zeros_like(E_wind)
    else:
        # 非节假日，根据具体 weather 判断
        if weather_value == 'cloudy':
            P_solar = np.zeros_like(P_solar)
            E_solar = np.zeros_like(E_solar)
        elif weather_value == 'windy':
            P_wind = np.zeros_like(P_wind)
            E_wind = np.zeros_like(E_wind)
        elif weather_value in ['rainy', 'snowy']:
            P_solar = np.zeros_like(P_solar)
            E_solar = np.zeros_like(E_solar)
            P_wind = np.zeros_like(P_wind)
            E_wind = np.zeros_like(E_wind)
        # 其余 weather 情况维持原状
    # ======================================================

    # ====== 叠加可再生能源功率并进行储能系统的充放电调度 ======
    P_renewable = P_solar + P_wind
    E_renewable = E_solar + E_wind

    E_max = 50000
    E_storage = np.zeros_like(t)
    E_storage[0] = E_max
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
        E_storage[i] = E_storage[i - 1]

        # 1) 先用太阳能满足
        E_solar_supply[i] = min(E_gen_solar, E_load_step)
        remaining_load = E_load_step - E_solar_supply[i]

        # 2) 再用风能满足
        E_wind_supply[i] = min(E_gen_wind, remaining_load)
        remaining_load -= E_wind_supply[i]

        # 当前时刻可再生能源多出的能量（可能为 0 或正数）
        net_renewable_energy = (E_gen_solar + E_gen_wind) - (E_solar_supply[i] + E_wind_supply[i])

        if net_renewable_energy > 0:
            # 可再生能源有富余 -> 给储能充电（上限 E_charge_max、SOC 上限 E_max）
            E_storage_charge_from_renewable[i] = min(net_renewable_energy, E_charge_max, E_max - E_storage[i])
            E_storage[i] += E_storage_charge_from_renewable[i]
            E_storage_charge_from_grid[i] = 0
            E_grid_supply[i] = 0
            E_storage_discharge[i] = 0
        else:
            # 可再生能源不够 -> 从储能放电，再不够就从电网补
            deficit_load = remaining_load
            E_storage_discharge[i] = min(deficit_load, E_discharge_max, E_storage[i])
            E_storage[i] -= E_storage_discharge[i]
            deficit_load -= E_storage_discharge[i]

            if deficit_load > 0:
                E_grid_supply[i] = deficit_load
                E_grid_draw[i] += E_grid_supply[i]

            # 判断是否需要从电网适度充电以维持目标 SOC
            if E_storage[i] < E_target:
                E_storage_charge_from_grid[i] = min(E_charge_max, E_target - E_storage[i])
                E_storage[i] += E_storage_charge_from_grid[i]
                E_grid_draw[i] += E_storage_charge_from_grid[i]
            else:
                E_storage_charge_from_grid[i] = 0

            E_storage_charge_from_renewable[i] = 0

        E_storage[i] = min(max(E_storage[i], 0), E_max)

    # ====== 结果统计 ======
    total_pv_supply_total = E_solar_supply.sum()
    total_wind_supply_total = E_wind_supply.sum()
    E_storage_discharge_total = E_storage_discharge.sum()
    E_grid_supply_total = E_grid_supply.sum()
    E_storage_charge_from_renewable_total = E_storage_charge_from_renewable.sum()
    E_storage_charge_from_grid_total = E_storage_charge_from_grid.sum()

    avg_v_wind = np.mean(v_wind)
    avg_wind_direction = np.mean(wind_direction)

    return (T_end,
            total_pv_supply_total,
            total_wind_supply_total,
            E_storage_discharge_total,
            E_grid_supply_total,
            E_storage_charge_from_renewable_total,
            E_storage_charge_from_grid_total,
            avg_v_wind,
            avg_wind_direction)

# ====== 主循环，对 load_data_df 逐行计算结果 ======
results = []
for idx, row in load_data_df.iterrows():
    (T_end,
     total_pv_supply_total,
     total_wind_supply_total,
     E_storage_discharge_total,
     E_grid_supply_total,
     E_storage_charge_from_renewable_total,
     E_storage_charge_from_grid_total,
     avg_v_wind,
     avg_wind_direction) = calculate_total_energy(row)

    results.append({
        "T_end": T_end,
        "total_pv_supply_total": total_pv_supply_total,
        "total_wind_supply_total": total_wind_supply_total,
        "E_storage_discharge_total": E_storage_discharge_total,
        "E_grid_supply_total": E_grid_supply_total,
        "E_storage_charge_from_renewable_total": E_storage_charge_from_renewable_total,
        "E_storage_charge_from_grid_total": E_storage_charge_from_grid_total,
        "avg_v_wind": avg_v_wind,
        "avg_wind_direction": avg_wind_direction
    })

output_df = pd.DataFrame(results)
output_path = "/Users/ethanzhu/Desktop/output_power.csv"
output_df.to_csv(output_path, index=False)
print(f"已将结果保存到 {output_path}")
import numpy as np
import heapq
import pandas as pd

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

    def dijkstra(energy_matrix, start_point, end_point):
        num_nodes = energy_matrix.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        distance = np.full(num_nodes, np.inf)
        previous = np.full(num_nodes, np.nan)
        distance[start_point] = 0

        # 使用优先队列来加速最短路径查找
        pq = [(0, start_point)]  # (distance, node)

        while pq:
            current_dist, current = heapq.heappop(pq)
            if visited[current]:
                continue
            visited[current] = True

            if current == end_point:
                break

            for neighbor in range(num_nodes):
                if not visited[neighbor] and energy_matrix[current, neighbor] < np.inf:
                    new_distance = current_dist + energy_matrix[current, neighbor]
                    if new_distance < distance[neighbor]:
                        distance[neighbor] = new_distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_distance, neighbor))

        # 构造最短路径
        min_path = []
        current = end_point
        while not np.isnan(current):
            min_path.insert(0, int(current) + 1)  # 转换为 1 索引
            current = previous[int(current)]

        E_total = distance[end_point]
        if np.isinf(E_total):
            min_path = []

        return min_path, E_total


    def calculate_energy_matrix(yard_matrix, Q_list_regular, P_list_regular, Q_total, special_connections, material_type):
        n = yard_matrix.shape[0]
        energy_matrix = np.full((n, n), np.inf)  # 初始化为无穷大，表示不可通行
        edge_indices = []
        regular_idx = 0
        special_conn = special_connections.get(material_type, [])

        # 使用字典来加速查找特定边
        special_edge_dict = {(sc['nodes'][0] - 1, sc['nodes'][1] - 1): sc for sc in special_conn}

        for i in range(n):
            for j in range(i + 1, n):
                if yard_matrix[i, j] == 1:
                    # 优先查找是否为特殊边
                    special_key = (i, j) if (i, j) in special_edge_dict else (j, i)
                    if special_key in special_edge_dict:
                        sc = special_edge_dict[special_key]
                        Q_value = sc['Q']
                        P_value = sc['P']
                        if Q_value == 0:
                            print(f"节点 {i+1} 和 {j+1} 之间的 Q 为 0，无法运输，设置为不可通行。")
                            continue
                        T_i = Q_total / Q_value
                        if P_value == 0:
                            print(f"节点 {i+1} 和 {j+1} 之间的 P 为 0，假设该连接不消耗能量。")
                            P_value = 1e-6  # 避免除零 

                        # 检查无效值
                        if np.isnan(P_value) or np.isnan(T_i) or np.isinf(P_value) or np.isinf(T_i):
                            print(f"无效的 P 或 T_i 值：P={P_value}, T_i={T_i}，发生在节点 {i+1} 到 {j+1}")
                            continue

                        energy_matrix[i, j] = energy_matrix[j, i] = P_value * T_i
                        edge_info = {'nodes': [i + 1, j + 1], 'Q': Q_value, 'P': P_value}
                        edge_indices.append(edge_info)
                    else:
                        if regular_idx >= len(Q_list_regular) or regular_idx >= len(P_list_regular):
                            raise ValueError('Q_list_regular 或 P_list_regular 的长度不足，无法匹配所有常规连接。')

                        Q_value = Q_list_regular[regular_idx]
                        P_value = P_list_regular[regular_idx]
                        regular_idx += 1

                        # 处理 Q 为 0 或无穷大的情况
                        if Q_value == 0 or np.isclose(Q_value, 0) or np.isinf(Q_value):
                            continue  # 跳过该连接
                        T_i = Q_total / Q_value

                        # 处理 P 为 0 或无穷大的情况
                        if P_value == 0 or np.isclose(P_value, 0) or np.isinf(P_value):
                            continue
                        # 检查无效值
                        if np.isnan(P_value) or np.isnan(T_i) or np.isinf(P_value) or np.isinf(T_i):
                            print(f"无效的 P_value 或 T_i 值，发生在节点 {i+1} 到 {j+1}。P_value: {P_value}, T_i: {T_i}")
                            continue

                        energy_matrix[i, j] = energy_matrix[j, i] = P_value * T_i
                        edge_info = {'nodes': [i + 1, j + 1], 'Q': Q_value, 'P': P_value}
                        edge_indices.append(edge_info)

        return energy_matrix, edge_indices


    def calculate_energy_consumption(Q_total, material_type, start_point, end_point):
        Ei = None
        # 定义三个堆场的邻接矩阵
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
        # 定义每个堆场的 Q_list（运量） 和 P_list（功率）常规
        Q_list_mineral_1 = np.concatenate([5000 * np.ones(7), 3500 * np.ones(2), 5000 * np.ones(14)])
        P_list_mineral_1 = np.concatenate([520 * np.ones(7), 90 * np.ones(2), 520 * np.ones(14)])
        Q_list_mineral_2 = np.concatenate([2500 * np.ones(15), [np.inf], 2500 * np.ones(3), [np.inf], 2500 * np.ones(3)])
        P_list_mineral_2 = np.concatenate([100 * np.ones(15), [np.inf], 100 * np.ones(3), [np.inf], 100 * np.ones(3)])
        Q_list_coal = np.concatenate([4500 * np.ones(26), [np.inf], 4100 * np.ones(5)])
        P_list_coal = np.concatenate([700 * np.ones(26), [np.inf], 644 * np.ones(5)])

        # 根据 material_type 选择堆场
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
        if end_point < 1 or end_point > yard_matrix.shape[0]:
            raise ValueError('起始点或终止点超出矩阵范围！')

        # 定义特殊连接的信息
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
        energy_matrix, edge_indices = calculate_energy_matrix(yard_matrix, Q_list_regular, P_list_regular, Q_total, special_connections, material_type)
        min_path, E_total = dijkstra(energy_matrix, start_point - 1, end_point - 1)

        # 计算 Ei
        if material_type == 'mineral_1':
            Ei = E_total + Q_total / 5000 * 1065 + Q_total / 5000 * 355
        elif material_type == 'mineral_2':
            Ei = E_total + Q_total / 5000 * 1065 + Q_total / 5000 * 355
        elif material_type == 'coal':
            Ei = E_total + Q_total / 5000 * 1065 + Q_total / 5000 * 355 + Q_total / 4500 * 700
        
        return Ei
    
    Ei = calculate_energy_consumption(Q_total, material_type, start_point, end_point)

    return Ei

results = []

for idx, row in load_data_df.iterrows():
    Ei = calculate_total_energy(row)  

    results.append({
        "Ei": Ei
    })

# 把 results 做成新的 DataFrame
output_df = pd.DataFrame(results)

output_path = "/Users/ethan/Desktop/output_Ei.csv"
output_df.to_csv(output_path, index=False)  

print(f"已将结果保存到 {output_path}")
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import weibull_min


## main
latitude = None
day_of_year = None
edge_indices = None
energy_matrix = None

mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 22,          # Global default font size
    'axes.labelsize': 22,     # Axis label font size
    'axes.titlesize': 24,     # Chart title font size
    'xtick.labelsize': 20,    # x-axis tick label size
    'ytick.labelsize': 20     # y-axis tick label size
})

# 获取用户输入
start_point = 1
end_point = int(input("请输入终止点： "))
Q_total = float(input("请输入总运量（吨）： "))
day_of_year = int(input("请输入今天是一年中的第几天（如6月21日是172）： "))
latitude = 36  # 地理纬度（度），可根据实际情况调整

# 获取材料类型并验证
material_type = input("请输入材料类型 (mineral_1, mineral_2, 或 coal): ")

while material_type not in ['mineral_1', 'mineral_2', 'coal']:
    print("输入无效。请输入 mineral_1, mineral_2, 或 coal。")
    material_type = input("请重新输入材料类型: ")

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
                        print(f"节点 {i+1} 和 {j+1} 之间的 Q 为 0 或无穷大，跳过该连接。")
                        continue  # 跳过该连接
                    T_i = Q_total / Q_value

                    # 处理 P 为 0 或无穷大的情况
                    if P_value == 0 or np.isclose(P_value, 0) or np.isinf(P_value):
                        print(f"节点 {i+1} 和 {j+1} 之间的 P 为 0 或无穷大，跳过该连接。")
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
    if start_point < 1 or start_point > yard_matrix.shape[0] or end_point < 1 or end_point > yard_matrix.shape[0]:
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

    Ep = Ei * 0.1229 / 1000
    Eb = (Q_total * 0.04 + Q_total * 0.05) * 0.83 * 1.4571 / 1000

    print(f'皮带总能耗为: {Ep:.2f} 吨标煤')
    print(f'搬倒总能耗为: {Eb:.2f} 吨标煤')

    return min_path, Ei, edge_indices, energy_matrix

# 调用计算最小能耗路径的函数
min_path, Ei, edge_indices, energy_matrix = calculate_energy_consumption(Q_total, material_type, start_point, end_point)

# 输出计算结果
print(f"最小路径: {min_path}")
print(f"总能耗: {Ei}")


# 初始化参数
start_time = 5
dt = 0.05
num_segments = len(min_path) - 1
# 预分配 segment_info
segment_info = [{'start_time': None, 'end_time': None, 'P_edge': None} for _ in range(num_segments)]
# 初始化最小输送能力
Q_min_edge = np.inf
total_P_edge = 0

# 遍历最小能耗路径上的每一段
for k in range(num_segments):
    node_start = min_path[k]
    node_end = min_path[k + 1]

    # 查找对应的边
    edge_found = False
    for idx in range(len(edge_indices)):
        nodes = edge_indices[idx]['nodes']
        if (nodes[0] == node_start and nodes[1] == node_end) or (nodes[0] == node_end and nodes[1] == node_start):
            Q_edge = edge_indices[idx]['Q']
            P_edge = edge_indices[idx]['P']
            edge_found = True
            break

    if not edge_found:
        raise ValueError(f'在 edge_indices 中未找到边 ({node_start}, {node_end})。可能原因：该边不可通行或数据有误。')

    Q_min_edge = min(Q_min_edge, Q_edge)
    total_P_edge += P_edge
    segment_info[k]['P_edge'] = P_edge

T_total = Q_total / Q_min_edge
print(f'最小输送能力为 {Q_min_edge:.2f} 吨/小时。')
T_end = T_total
t = np.arange(start_time, start_time + T_end, dt)
E_load = np.full_like(t, total_P_edge)
print(f'运输任务在 {T_end:.2f} 小时内完成。')




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

P_wind *= N_wind_turbine
E_wind = P_wind * dt

P_renewable = P_solar + P_wind
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

# 绘图
def plot_results(t, E_solar_supply, E_wind_supply, E_storage_discharge, E_grid_supply, E_solar, E_wind, E_load, E_storage, E_max, material_type, start_time, T_end):
    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    
    # (1) Energy supply sources stack plot
    axs[0].stackplot(t, E_solar_supply, E_wind_supply, E_storage_discharge, E_grid_supply, labels=['Solar Power Supply', 'Wind Power Supply', 'Storage Discharge Supply', 'Grid Power Supply'])
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Energy (kWh)')
    axs[0].set_title('Energy Supply Sources Stack Plot')
    # Move legend to upper right corner, outside the plot
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1), framealpha=0.9, edgecolor="black")
    axs[0].grid(True)

    # (2) Supply curves for different energy types
    axs[1].plot(t, E_solar_supply, label='Solar Power Supply', linestyle='-')
    axs[1].plot(t, E_wind_supply, label='Wind Power Supply', linestyle='--')
    axs[1].plot(t, E_storage_discharge, label='Storage Discharge', linestyle='-.')
    axs[1].plot(t, E_grid_supply, label='Grid Power Supply', linestyle=':')
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Power Supply (kW)')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Energy Supply over Time')
    axs[1].grid(True)

    # (3) Renewable energy generation vs load demand curve
    axs[2].plot(t, E_solar + E_wind, 'g--', label='Total Solar and Wind Generation')
    axs[2].plot(t, E_load * (t[1] - t[0]), 'm:', label='Load Energy Demand')
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel('Energy (kWh)')
    axs[2].set_title('Renewable Energy Generation vs Load Demand')
    # Move legend to upper right corner, outside the plot
    axs[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1), framealpha=0.9, edgecolor="black")
    axs[2].grid(True)

    # (4) Storage State of Charge (SOC) curve
    axs[3].plot(t, E_storage / E_max * 100, 'c', linewidth=2)
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel('State of Charge (%)')
    axs[3].set_title('Storage System SOC Over Time')
    axs[3].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Pie chart for energy source contribution at a specific time point
    t0 = start_time + T_end / 2  # Any time point
    idx = np.argmin(np.abs(t - t0))
    energy_sources = [E_solar_supply[idx], E_wind_supply[idx], E_storage_discharge[idx], E_grid_supply[idx]]
    total_supply = sum(energy_sources)
    energy_percentage = np.array(energy_sources) / total_supply * 100 if total_supply > 0 else np.zeros_like(energy_sources)
    energy_labels = ['Solar Power Supply', 'Wind Power Supply', 'Storage Supply', 'Grid Supply']
    energy_labels_with_percent = [f'{lbl}: {pct:.1f}%' for lbl, pct in zip(energy_labels, energy_percentage)]

    # Create a new figure for the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(energy_sources, labels=energy_labels_with_percent, autopct='%1.1f%%', startangle=90)
    plt.title(f'{material_type} Energy Contribution at {t[idx]:.2f} Hours')

    plt.show()

plot_results(
    t, 
    E_solar_supply, 
    E_wind_supply, 
    E_storage_discharge, 
    E_grid_supply, 
    E_solar, 
    E_wind, 
    E_load, 
    E_storage, 
    E_max, 
    material_type,
    start_time,
    T_end
)
import numpy as np
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


## main
latitude = None
day_of_year = None
Ei = None
edge_indices = None
energy_matrix = None

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




def calculate_energy_consumption(Q_total, material_type, start_point, end_point):
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

    return min_path, E_total, Ei, edge_indices, energy_matrix

def dijkstra(energy_matrix, start_point, end_point):
    num_nodes = energy_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)
    distance = np.full(num_nodes, np.inf)  # 初始化距离为无穷大
    previous = np.full(num_nodes, np.nan)
    distance[start_point] = 0

    for _ in range(num_nodes):
        unvisited_indices = np.where(~visited)[0]
        if len(unvisited_indices) == 0:
            break
        min_idx = np.argmin(distance[unvisited_indices])
        current = unvisited_indices[min_idx]
        visited[current] = True

        if current == end_point:
            break

        for neighbor in range(num_nodes):
            if not visited[neighbor] and energy_matrix[current, neighbor] < np.inf:
                new_distance = distance[current] + energy_matrix[current, neighbor]
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    previous[neighbor] = current

    min_path = []
    current = end_point
    while not np.isnan(current):
        min_path.insert(0, int(current) + 1)  # 转换为 1 索引
        current = previous[int(current)]

    E_total = distance[end_point]
    if np.isinf(E_total):
        min_path = []

    return min_path, E_total



# 调用计算最小能耗路径的函数
min_path, E_total, Ei, edge_indices, energy_matrix = calculate_energy_consumption(Q_total, material_type, start_point, end_point)

def calculate_energy_matrix(yard_matrix, Q_list_regular, P_list_regular, Q_total, special_connections, material_type):
    n = yard_matrix.shape[0]
    energy_matrix = np.full((n, n), np.inf)
    edge_indices = []

    regular_idx = 0
    special_conn = special_connections.get(material_type, [])

    for i in range(n):
        for j in range(i + 1, n):
            if yard_matrix[i, j] == 1:
                is_special = False
                for sc in special_conn:
                    if (sc['nodes'][0] == i and sc['nodes'][1] == j) or (sc['nodes'][0] == j and sc['nodes'][1] == i):
                        is_special = True
                        edge_info = {'nodes': [i + 1, j + 1], 'Q': sc['Q'], 'P': sc['P']}
                        T_i = Q_total / sc['Q']
                        energy_matrix[i, j] = energy_matrix[j, i] = sc['P'] * T_i
                        edge_indices.append(edge_info)
                        break

                if not is_special:
                    if regular_idx >= len(Q_list_regular) or regular_idx >= len(P_list_regular):
                        raise ValueError('Q_list_regular 或 P_list_regular 的长度不足，无法匹配所有常规连接。')

                    Q_value = Q_list_regular[regular_idx]
                    P_value = P_list_regular[regular_idx]
                    regular_idx += 1

                    T_i = Q_total / Q_value
                    energy_matrix[i, j] = energy_matrix[j, i] = P_value * T_i
                    edge_info = {'nodes': [i + 1, j + 1], 'Q': Q_value, 'P': P_value}
                    edge_indices.append(edge_info)

    return energy_matrix, edge_indices

# 输出计算结果
print(f"最小路径: {min_path}")
print(f"总能耗: {E_total}")
print(f"Ei: {Ei}")
print(f"边索引: {edge_indices}")
print(f"能耗矩阵: \n{energy_matrix}")


# 初始化参数
start_time = 5
dt = 0.05  # 时间步长（小时）
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

    # 更新最小输送能力
    Q_min_edge = min(Q_min_edge, Q_edge)

    # 累加总的功率需求
    total_P_edge += P_edge

    # 存储该段的信息
    segment_info[k]['P_edge'] = P_edge

# 计算总的运输时间，由最小的输送能力决定
T_total = Q_total / Q_min_edge

# 输出最小输送能力
print(f'最小输送能力为 {Q_min_edge:.2f} 吨/小时。')

# 设置模拟总时间为运输任务的总时间
T_end = T_total

# 生成时间向量（实际时间）
t = np.arange(start_time, start_time + T_end, dt)

# 初始化负荷曲线
E_load = np.full_like(t, total_P_edge)

# 输出运输任务完成时间
print(f'运输任务在 {T_end:.2f} 小时内完成。')

# 1. 计算日出和日落时间
day_of_year = 172  # 可根据需要设置
latitude = 30  # 设置纬度

declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
hour_angle = np.degrees(np.arccos(-np.tan(np.deg2rad(latitude)) * np.tan(np.deg2rad(declination))))

sunrise = 12 - hour_angle / 15
sunset = 12 + hour_angle / 15

sunrise = max(0, sunrise)
sunset = min(24, sunset)

print(f'当天日出时间：{sunrise:.2f} 点')
print(f'当天日落时间：{sunset:.2f} 点')

# 2. 太阳能发电参数
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

# 3. 风能发电参数
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

# 4. 总可再生能源发电功率
P_renewable = P_solar + P_wind
E_renewable = E_solar + E_wind

# 5. 分布式能量储存参数
E_max = 50000  # 储能系统最大容量（kWh）
E_storage = np.zeros_like(t)
E_storage[0] = E_max  # 初始储能水平

P_charge_max = 1000
P_discharge_max = P_charge_max

E_charge_max = P_charge_max * dt
E_discharge_max = P_discharge_max * dt

E_target = E_max * 0.8

# 6. 能量调度
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

# 7. 绘制结果

# （1）能源供应来源堆叠图
plt.figure()
plt.stackplot(t, E_solar_supply, E_wind_supply, E_storage_discharge, E_grid_supply, labels=['太阳能供电', '风能供电', '储能放电供电', '电网供电'])
plt.xlabel('时间（小时）')
plt.ylabel('能量（kWh）')
plt.title('能源供应来源堆叠图')
plt.legend()
plt.grid(True)
plt.show()

# （2）各能源类型时序供应曲线
plt.figure()
plt.plot(t, E_solar_supply, 'g', label='太阳能供电')
plt.plot(t, E_wind_supply, 'c', label='风能供电')
plt.plot(t, E_storage_discharge, 'm', label='储能放电供电')
plt.plot(t, E_grid_supply, 'r', label='电网供电')
plt.xlabel('时间（小时）')
plt.ylabel('能量（kWh）')
plt.title('各能源类型时序供应曲线')
plt.legend()
plt.grid(True)
plt.show()

# （3）可再生能源和负荷需求关系曲线图
plt.figure()
plt.plot(t, E_solar + E_wind, 'g--', label='光伏、风能总发电量')
plt.plot(t, E_load * dt, 'm:', label='负荷能耗需求')
plt.xlabel('时间（小时）')
plt.ylabel('能量（kWh）')
plt.title('光伏、风能发电与负荷需求关系图')
plt.legend()
plt.grid(True)
plt.show()

# （4）储能水平变化图（SOC）
plt.figure()
plt.plot(t, E_storage / E_max * 100, 'c', linewidth=2)
plt.xlabel('时间（小时）')
plt.ylabel('储能水平（百分比）')
plt.title('储能设备 SOC（百分比）随时间变化图')
plt.grid(True)
plt.show()

# （5）在特定时间点各能源对负荷的供应比例图
t0 = start_time + T_end / 2
idx = np.argmin(np.abs(t - t0))

energy_sources = [E_solar_supply[idx], E_wind_supply[idx], E_storage_discharge[idx], E_grid_supply[idx]]
total_supply = sum(energy_sources)
energy_percentage = np.array(energy_sources) / total_supply * 100 if total_supply > 0 else np.zeros_like(energy_sources)

energy_labels = ['太阳能供电', '风能供电', '储能供电', '电网供电']
energy_labels_with_percent = [f'{lbl}: {pct:.1f}%' for lbl, pct in zip(energy_labels, energy_percentage)]

plt.figure()
plt.pie(energy_sources, labels=energy_labels_with_percent, autopct='%1.1f%%')
plt.title(f'{material_type} {t[idx]:.2f} 小时各能源对负荷的供应比例')
plt.show()
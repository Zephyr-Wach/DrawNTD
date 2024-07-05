import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform

# 定义并加载中文字体
font_path = './Data/txt.ttf'
prop = FontProperties(fname=font_path)

# 读取数据集
data_set_path = './Data/DataSet.xlsx' #数据通讯记录
device_list_path = './Data/deviceList.xlsx' #设备列表
data = pd.read_excel(data_set_path, index_col=0)
device_info = pd.read_excel(device_list_path)

# 解析设备列表，生成设备名到IP地址的映射
device_map = {}
for _, row in device_info.iterrows():
    device_name = row['设备名']
    ip_range = row['ip地址']
    if '/' in ip_range:
        base_ip, *offsets_str = ip_range.split('/')
        base_ip_parts = base_ip.split('.')
        offsets = [int(o) for o in offsets_str]
        for offset in offsets:
            full_ip = f"{base_ip_parts[0]}.{base_ip_parts[1]}.{base_ip_parts[2]}.{offset}"
            device_map[full_ip] = device_name
    elif '-' in ip_range:
        start_ip, end_ip = ip_range.split('-')
        start, end = map(int, (start_ip.split('.')[-1], end_ip))
        for i in range(start, end + 1):
            full_ip = f"172.20.0.{i}"
            device_map[full_ip] = device_name

# 创建空图
G = nx.Graph()

# 添加边
for row in data.itertuples():
    source_ip, target_ip = str(row[1]), str(row[2])
    G.add_edge(source_ip, target_ip)

# 确定颜色映射
unique_devices = set(device_map.values())
color_map = {device: f"C{i}" for i, device in enumerate(unique_devices)}

# 未知设备默认颜色（灰色）和对应的标签（未知）
default_color = "gray"
unknown_label = "未知"

# 分配节点颜色
node_colors = [
    color_map.get(device_map.get(ip, unknown_label), default_color)
    for ip in G.nodes()
]

# 计算节点度数
degrees = dict(G.degree())

# 确定核心节点数量
top_n = 6
core_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]

# 创建子图和布局
sub_graphs = {}
sub_pos = {}

# 新增部分：为核心节点设定初始的环形布局
radius = 1.0  # 环形半径
angle_step = (2 * math.pi) / top_n  # 每个节点间的角度差
core_node_angles = [i * angle_step for i in range(top_n)]
core_node_pos_centered = {node: (radius * math.cos(angle), radius * math.sin(angle)) for node, angle in zip(core_nodes, core_node_angles)}

# 使用核心节点的环形布局作为初始位置
for core_node, init_pos in core_node_pos_centered.items():
    sub_pos[core_node] = {core_node: init_pos}

# 应用布局到其他节点
for core_node in core_nodes:
    sub_graphs[core_node] = G.subgraph([core_node] + list(G.neighbors(core_node)))
    sub_pos[core_node] = nx.spring_layout(sub_graphs[core_node], seed=42, pos=sub_pos[core_node])

# 合并所有子图的布局
pos = {}
for core_node, p in sub_pos.items():
    for node, xy in p.items():
        pos[node] = xy

# 调整过近的节点位置
nodes_positions = np.array([pos[node] for node in G.nodes()])
distances = squareform(pdist(nodes_positions))
threshold = 1.8

# 调整过近的节点
for i in range(len(G.nodes())):
    for j in range(i+1, len(G.nodes())):
        if distances[i, j] < threshold:
            offset_direction = nodes_positions[j] - nodes_positions[i]
            offset_magnitude = 0.05 * np.random.rand()
            offset = offset_magnitude * offset_direction / np.linalg.norm(offset_direction)
            nodes_positions[j] += offset

# 更新 pos 字典以反映新的节点位置
for i, node in enumerate(G.nodes()):
    pos[node] = tuple(nodes_positions[i])

# 绘制网络图，使用设备名对应的颜色
nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=node_colors, node_size=100)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=6)

# 添加图例
legend_handles = [mpatches.Patch(color=color, label=device) for device, color in color_map.items()]
plt.legend(
    handles=legend_handles,
    loc='lower left',
    bbox_to_anchor=(0, 0),
    prop=prop,
    fontsize=6,
    ncol=2,
    handlelength=1,
    handletextpad=0.5,
    borderaxespad=0.5,
    scatterpoints=1
)
# 在图例处理中，"未知"也被包含进去
if unknown_label in color_map.values():
    legend_handles.append(mpatches.Patch(color=default_color, label=unknown_label))

# 显示图形
plt.axis('off')
plt.show()
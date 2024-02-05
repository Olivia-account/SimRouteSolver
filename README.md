# SimRouteSolver

# 项目介绍
路径规划之所以成为物流和运输中的关键问题，是因为它直接影响着资源利用效率和服务质量。在城市交通、货物配送等领域，如何科学合理地规划路径，使得车辆行驶最短，成为了迫切需要解决的问题。传统的算法可能难以应对复杂多变的实际情况，因此我们选择了模拟退火算法，这种模仿金属冶炼中的退火过程的算法，具有全局搜索的优势，能够更好地适应各种实际场景。
# 关键词
1. 路径规划
2. 模拟退火算法
3. 物流优化
4. 车辆路径
5. 智能算法
# 效果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c4a34605ea0f4209b60c14d2d43caabb.png)

# 项目思路
整个项目的实现思路可以分为以下步骤：
1. 数据读取和处理：
  - 通过read_excel函数读取Excel文件，获取节点信息和距离矩阵。
  - 初始化模型，包括节点集合、起点（配送中心）、车辆容量等信息。
2. 初始解生成：
  - 使用initial_sol函数生成随机的初始解，即节点的初始访问顺序。
3. 模拟退火算法主循环：
  - 初始化模拟退火算法参数，如初始温度（T0）、最终温度（Tf）、冷却因子（detaT）等。
  - 进入主循环，直至温度降至最终温度。
  - 在每个温度下进行局部搜索，通过随机选择交换、逆转或插入操作生成新解。
  - 判断是否接受新解，根据模拟退火概率进行决策。
4. 路径分割和距离计算：
  - 使用splitRoutes函数将TSP解分割成车辆路径，计算车辆数。
  - 利用calDistance函数计算路径的行驶距离。
5. 优化结果记录和输出：
  - 记录每次迭代中的最优解。
  - 输出最终优化的路径、车辆数以及行驶距离。
6. 效果展示：
  - 使用plotObj函数绘制优化过程中目标函数值的变化图。
# 具体代码
代码注释解释：
1. num_vehicle: 记录车辆数量，初始为1。
2. vehicle_routes: 存储所有车辆路径的列表。
3. route: 存储当前车辆路径的列表。
4. remained_cap: 记录当前车辆路径的剩余容量。
5. 遍历节点序列，判断节点是否超出车辆容量。
6. 如果未超出容量，将节点加入当前车辆路径，并更新剩余容量。
7. 如果超出容量，将当前车辆路径添加到车辆路径列表，重新初始化当前路径，并增加车辆数量。
8. 返回车辆数量和车辆路径列表。
整体作用解释：
该函数的作用是将TSP（Traveling Salesman Problem，旅行商问题）的解分割成为车辆路径，同时计算车辆数量。在物流领域中，TSP解表示了一组节点的访问顺序，而车辆路径则是根据容量限制将这些节点分配给不同的车辆，确保每辆车的容量不超过设定值。函数返回车辆数量和车辆路径的列表，为后续计算路径距离和优化结果提供基础。

```python
# 将TSP分割成为车辆路径，并计算车辆数
def splitRoutes(nodes_seq, model):
    # 初始化车辆数量为1
    num_vehicle = 1
    # 存储车辆路径的列表
    vehicle_routes = []
    # 存储当前车辆路径的列表
    route = []
    # 剩余车辆容量，初始值为车辆容量
    remained_cap = model.vehicle_cap
    
    # 遍历节点序列
    for node_no in nodes_seq:
        # 判断当前节点是否超出车辆容量
        if remained_cap - model.node_list[node_no].demand >= 0:
            # 如果未超出容量，将节点加入当前车辆路径
            route.append(node_no)
            # 更新剩余车辆容量
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            # 如果超出容量，将当前车辆路径添加到车辆路径列表，并重新初始化当前路径
            vehicle_routes.append(route)
            route = [node_no]
            # 增加车辆数量
            num_vehicle = num_vehicle + 1
            # 重置剩余车辆容量
            remained_cap = model.vehicle_cap - model.node_list[node_no].demand
    
    # 将最后一个车辆路径添加到路径列表
    vehicle_routes.append(route)
    
    # 返回车辆数量和车辆路径列表
    return num_vehicle, vehicle_routes

```
代码注释解释：
1. distance: 用于存储路径的总距离。
2. depot: 获取模型中的配送中心信息。
3. 遍历车辆路径中的节点，计算节点之间的距离。
4. from_node和to_node: 获取当前节点和下一个节点的信息。
5. 使用距离矩阵（matrix）中的对应值，累加节点之间的距离。
6. 计算首尾节点与配送中心的距离并累加。
7. 返回路径的总距离。
整体作用解释：
该函数用于计算给定车辆路径的总行驶距离。通过遍历路径中的节点，累加节点之间以及首尾节点与配送中心之间的距离，得到整个路径的行驶距离。这个距离是评估路径优劣的指标之一，在路径规划问题中，优化这个距离即是优化路径的目标。函数返回路径的总距离，为后续评估和优化提供了基础。

```python
# 计算路径距离
def calDistance(route, model):
    # 初始化路径距离
    distance = 0
    # 获取配送中心信息
    depot = model.depot
    
    # 遍历车辆路径中的节点，计算节点之间的距离
    for i in range(len(route) - 1):
        # 获取当前节点和下一个节点
        from_node = model.node_list[route[i]]
        to_node = model.node_list[route[i + 1]]
        
        # 累加节点之间的距离，使用距离矩阵中的对应值
        distance += matrix.iloc[from_node.seq_no + 1, to_node.seq_no + 1]
    
    # 计算首尾节点与配送中心的距离并累加
    first_node = model.node_list[route[0]]
    last_node = model.node_list[route[-1]]
    distance += matrix.iloc[0, first_node.seq_no + 1]
    distance += matrix.iloc[0, last_node.seq_no + 1]
    
    # 返回路径的总距离
    return distance

```
# 项目链接
CSDN：https://blog.csdn.net/m0_46573428/article/details/136044851
# 后记
如果觉得有帮助的话，求 关注、收藏、点赞、星星 哦！

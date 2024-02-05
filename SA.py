# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
import random
import time
import copy
import math
import sys
np.set_printoptions(threshold = np.inf)

class Logger(object):
    def __init__(self, filename = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
class Sol():
    def __init__(self):
        self.nodes_seq = None  # TSP解
        self.obj = None  # 优化目标值之车辆数量
        self.dis = None  # 优化行驶距离
        self.routes = None  # VRP解


class Node():
    def __init__(self):
        self.name = 0  # 节点名称，对应gis农田ObjectID
        self.seq_no = 0  # 节点编号,从0开始
        self.demand = 0  # 节点需求量


class Model():
    def __init__(self):
        self.best_sol = None  # 全局最优解
        self.node_list = []  # 节点集合
        self.node_seq_no_list = []  # 节点编号集合
        self.depot = None  # 配送中心
        self.number_of_nodes = 0  # 需求点数量
        self.vehicle_cap = 25  # 车辆容量
        
# 读取数据
def read_excel(filepath, model, sheet_name=''):
    seq = -1
    df = pd.read_excel(filepath)
    global matrix
    matrix = df.iloc[:, 1:-1]
    for i in range(df.shape[0]):
        node = Node()
        node.seq_no = seq
        node.name = df['no'][i]
        node.demand = df['demand'][i]/0.5048
        if df['demand'][i] == 0:
            model.depot = node
        else:
            model.node_list.append(node)
            model.node_seq_no_list.append(node.seq_no)
        seq = seq + 1
    model.number_of_nodes = len(model.node_list)
    print('数据读取操作完成')

# 随机生成初始解
def initial_sol(node_seq):
    node_seq = copy.deepcopy(node_seq)
    random.seed(12)
    random.shuffle(node_seq)
    print("生成初始解为：", node_seq)
    return node_seq


# 交换操作
def swap(nodes_seq):
    sol_length = len(nodes_seq) - 1
    i = random.randint(0, sol_length)
    j = random.randint(0, sol_length)
    while i == j:
        j = random.randint(0, sol_length)
    nodes_seq[i], nodes_seq[j] = nodes_seq[j], nodes_seq[i]
    #print("交换操作后解为：", nodes_seq)
    return nodes_seq

# 逆转操作
def reverse(nodes_seq):
    sol_length = len(nodes_seq) - 1
    i = random.randint(0, sol_length)
    j = random.randint(0, sol_length)
    while i == j:
        j = random.randint(0, sol_length)
    if i < j:
        reverse_part = nodes_seq[i:j + 1]
        #print("逆转片段:", reverse_part)
        last_part = nodes_seq[j + 1:]
        del nodes_seq[i:]
        reverse_part.reverse()
        nodes_seq.extend(reverse_part)
        nodes_seq.extend(last_part)
    else:
        reverse_part = nodes_seq[j:i + 1]
        #print("逆转片段:", reverse_part)
        last_part = nodes_seq[i + 1:]
        del nodes_seq[j:]
        reverse_part.reverse()
        nodes_seq.extend(reverse_part)
        nodes_seq.extend(last_part)
    #print("逆转操作后解为：", nodes_seq)
    return nodes_seq

# 插入操作,将选择的第一个位置插入到选择的第二个位置后
def insert(nodes_seq):
    sol_length = len(nodes_seq) - 1
    i = random.randint(0, sol_length)
    selected_one = nodes_seq[i]
    #print("进行插入操作元素为", i, nodes_seq[i])
    del nodes_seq[i]
    j = random.randint(0, sol_length - 1)
    while j == i - 1:  # 保证i不被插入回原来的位置
        j = random.randint(0, sol_length - 1)
    #print("被插入位置为：", j, nodes_seq[j])
    nodes_seq.insert(j + 1, selected_one)
    #print("插入操作后解为：", nodes_seq)
    return nodes_seq


# 轮盘赌法选择领域操作
# 选择交换操作的概率[0,0.2),选择逆转操作的概率[0.2,0.7),选择插入操作的概率[0.7,1）
def roulette(p_swap = 0.2, p_reverse = 0.5, p_insert = 0.3):
    random_number = random.uniform(0, 1)  # 不包含1
    if 0 <= random_number < p_swap:
        way = 0  # swap
        #print("轮盘赌法选择领域操作为：交换")
    elif p_swap <= random_number < p_swap + p_reverse:
        way = 1  # reverse
        #print("轮盘赌法选择领域操作为：逆转")
    else:
        way = 2  # insert
        #print("轮盘赌法选择领域操作为：插入")
    return way

def cal_action(n):
    action_list = []
    for i in range(n):
        way = roulette(0.2, 0.5, 0.3)
        action_list.append(way)
    print("操作列表为：", action_list)
    return action_list


def action(nodes_seq, way):
    nodes_seq = copy.deepcopy(nodes_seq)
    if way == 0:
        nodes_seq = swap(nodes_seq)
        return nodes_seq
    elif way == 1:
        nodes_seq = reverse(nodes_seq)
        return nodes_seq
    else:
        nodes_seq = insert(nodes_seq)
        return nodes_seq
    
# 将TSP分割成为车辆路径，并计算车辆数
def splitRoutes(nodes_seq, model):
    num_vehicle = 1
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            route.append(node_no)
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle = num_vehicle + 1
            remained_cap = model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle, vehicle_routes

# 计算路径距离
def calDistance(route, model):
    distance = 0
    depot = model.depot
    for i in range(len(route) - 1):
        from_node = model.node_list[route[i]]
        to_node = model.node_list[route[i + 1]]
        distance += matrix.iloc[from_node.seq_no+1, to_node.seq_no+1]
    first_node = model.node_list[route[0]]
    last_node = model.node_list[route[-1]]
    distance += matrix.iloc[0, first_node.seq_no+1]
    distance += matrix.iloc[0, last_node.seq_no+1]
    return distance


# 计算目标函数
def calObj(nodes_seq, model):
    num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
    distances = 0
    for route in vehicle_routes:
        distances += calDistance(route, model)
    return num_vehicle, vehicle_routes, distances


# 作图
def plotObj(obj_list, pic_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.grid()
    plt.xlim(1, len(obj_list) + 1)
    plt.savefig(pic_path)
    # plt.show()
    

# SA主函数
# T0初始温度，Tf最终温度，deltaT冷却因子
def run(filepath, T0, Tf, detaT, xlsx_path, pic_path, sheet_name=''):
    model = Model()
    read_excel(filepath, model, sheet_name=sheet_name)
    history_best_dis = []
    sol = Sol()
    sol.nodes_seq = initial_sol(model.node_seq_no_list)
    sol.obj, sol.routes, sol.dis = calObj(sol.nodes_seq, model)
    print("初始解目标值为：", sol.obj, sol.routes, sol.dis)
    firsts = copy.deepcopy(sol.routes)
    for first in firsts:
        for i in range(len(first)):
            first[i] = model.node_list[first[i]].name
    work = xlsxwriter.Workbook(xlsx_path)
    worksheet = work.add_worksheet(sheet_name)
    worksheet.write(0, 0, 'filename')
    worksheet.write(0, 1, 'drive distance_first')
    worksheet.write(0, 2, 'num_vehicle_first')
    worksheet.write(0, 3, 'routes_first')
    worksheet.write(0, 4, 'drive distance_best')
    worksheet.write(0, 5, 'num_vehicle_best')
    worksheet.write(0, 6, 'routes_best')
    worksheet.write(1, 0, sheet_name)
    worksheet.write(1, 1, sol.dis)
    worksheet.write(1, 2, sol.obj)
    for row, route in enumerate(firsts):
        r = [str(i) for i in route]
        worksheet.write(row + 1, 3, '-'.join(r))
    model.best_sol = copy.deepcopy(sol)
    history_best_dis.append(sol.dis)
    Tk = T0  # 当前温度
    while Tk >= Tf:
        print("当前温度为：", Tk)
        action_list = cal_action(500)
        for i in range(500):
            new_sol = Sol()
            new_sol.nodes_seq = action(sol.nodes_seq, action_list[i])
            new_sol.obj, new_sol.routes, new_sol.dis = calObj(new_sol.nodes_seq, model)
            deta_f = new_sol.dis - sol.dis
            # New interpretation of acceptance criteria
            if deta_f < 0 or math.exp(-deta_f / Tk) > random.random():
                sol = copy.deepcopy(new_sol)
            if sol.dis < model.best_sol.dis:
                model.best_sol = copy.deepcopy(sol)

        Tk = Tk * detaT
        history_best_dis.append(model.best_sol.dis)
        print("当前温度下最优解为：", model.best_sol.dis)
    print("降温已完成--------------------------------------------------------------------------------")
    print("最优解优化过程：", history_best_dis)
    print("温度：%s，当前解:%s 最优解: %s" % (Tk, sol.dis, model.best_sol.dis))
    print("最优路径为：", model.best_sol.routes)
    print("优化比例：", history_best_dis[-1]/history_best_dis[0]*100, '%')
    for route in model.best_sol.routes:
        for i in range(len(route)):
            route[i] = model.node_list[route[i]].name
    print("最优路径为：", model.best_sol.routes)
    worksheet.write(1, 4, model.best_sol.dis)
    worksheet.write(1, 5, model.best_sol.obj)
    for row, route in enumerate(model.best_sol.routes):
        r = [str(i) for i in route]
        worksheet.write(row + 1, 6, '-'.join(r))
    work.close()
    plotObj(history_best_dis, pic_path)


import sys, os

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python SA.py <name>")
    #     sys.exit(1)
    print('kaishiyunxing1')

    name = sys.argv[1]
    # print("Received variable:", name)
    print('开始计算'+name)
    # 如果不需要避开已经计算完的，将这段代码注释
    # 检查文件数量是否为3
    if len(os.listdir('result/'+name)) == 3:
        print(name+'已经计算完毕')
    else:    
        start = time.perf_counter()
        print('正在计算：'+name)
        p = 'path/'
        r = 'result/'+name+'/'
        # sys.stdout = Logger(r + name +'.txt')
        sys.stdout = open(r + name +'.txt', mode='w', encoding='utf-8')

        file = p+name + '.xlsx'
        xlsx_path = r+'sa'+name+'.xlsx'
        pic_path = r+name+'.png'
        sheet_name = 'Sheet2'
        run(filepath = file, T0 = 1000, Tf = 0.001, detaT = 0.99, xlsx_path = xlsx_path, pic_path=pic_path, sheet_name=sheet_name)
        print("运行过程输出成功")
        end = time.perf_counter()
        print('用时：{:.4f}min'.format((end - start)/60))




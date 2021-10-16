#coding:utf-8
import networkx as nx
import numpy as np
from collections import defaultdict
import sys
import time
import threading

G = nx.Graph()
def constructGraph(filename):  
    with open(filename) as file:
        for line in file:
            line = line.strip('\n')
            head, tail, weight = [str(x) for x in line.split('\t')]
            G.add_weighted_edges_from([(head, tail, weight)])
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def k_shell(graph):
    G = graph;  degreeList = []; relations = defaultdict(list)
    list_k_shell_1 = sorted(nx.core_number(G).items(), key=lambda x: x[1]); switch = 'ON'
    for key in list_k_shell_1:
        if switch == 'ON':
            k_shell_value = key[1]; switch = 'OFF'
            relations[k_shell_value].append(key[0])
            continue
        if key[1] == k_shell_value:
            relations[k_shell_value].append(key[0])
        else:
            k_shell_value = key[1]
            relations[k_shell_value].append(key[0])
    return relations

def inflate(graph, subgraph_node):
    G = graph; SG_neighbor_nodes = []
    for value1 in subgraph_node:
        SG_neighbor_nodes.extend(G.neighbors(value1))
    SG_neighbor_nodes = list(set(SG_neighbor_nodes).difference(set(subgraph_node)))  #Get neighbor nodes of subgraph
    
    if len(subgraph_node) == 1:
        raw_fitness_SG = 0; raw_expectation_edge = 0
    else:
        inner_SG_weight, out_SG_weight, Modularity, edge_num = modularity(G,subgraph_node)     #Fitness score of initial subgraph
        density = weight_density(G,subgraph_node,inner_SG_weight, edge_num)
        raw_fitness_SG = (float(density) + float(Modularity)) / 2 + 1 / (1 / float(density) +1 / float(Modularity))
        raw_expectation_edge = float(raw_fitness_SG) * len(subgraph_node)
    
    while SG_neighbor_nodes:
        maxNode = ''
        for nei_node in SG_neighbor_nodes:
            neighbor_node = []; neighbor_node = G.neighbors(nei_node)
            list_contact_nodes = list(set(neighbor_node) & set(subgraph_node))
            edge_count = 0
            for value2 in list_contact_nodes:
                edge_wei = G.get_edge_data(nei_node, value2)["weight"] #The edge weights of neighbor nodes and subgraph nodes
                if float(edge_wei) > float(0.5):
                    edge_count = edge_count + 1
            actually_edges = len(list(set(neighbor_node) & set(subgraph_node)))
            subgraph_node.append(nei_node)   
            inner_SG_weight, out_SG_weight, Modularity, edge_num = modularity(G,subgraph_node)
            density = weight_density(G,subgraph_node,inner_SG_weight, edge_num)
            if (density != 0 or Modularity != 0):
                fitness_SG = (float(density) + float(Modularity)) / 2 + 1 / (1 / float(density) +1 / float(Modularity)) 
            else:
                fitness_SG = 0
            subgraph_node.remove(nei_node)
        
            if(fitness_SG > raw_fitness_SG and actually_edges >= raw_expectation_edge and edge_count >= 1):
                raw_fitness_SG = fitness_SG; raw_expectation_edge = actually_edges
                maxNode = nei_node; maxDensity = density

        if maxNode:
            subgraph_node.append(maxNode)
            SG_neighbor_nodes.remove(maxNode)
        else:
            break
    
    return subgraph_node, raw_fitness_SG, raw_expectation_edge

def shrink(graph, subgraph_node, raw_fitness_SG, raw_expectation_edge):
    G = graph; inner_node = []
    for value1 in subgraph_node:
        neighbor_node = [];  interaction_node = []
        neighbor_node = G.neighbors(value1)
        interaction_node = set(neighbor_node) & set(subgraph_node)
        if (len(list(neighbor_node)) != len(list(interaction_node))):
            inner_node.append(value1)
        else:
            continue
    
    while inner_node:
        minNode = ''
        for in_node in inner_node:
            neighbor_node = []; neighbor_node = G.neighbors(in_node)
            actually_edges = len(list(set(neighbor_node) & set(subgraph_node)))
            subgraph_node.remove(in_node)   
            inner_SG_weight, out_SG_weight, Modularity, edge_num = modularity(G,subgraph_node)
            density = weight_density(G,subgraph_node,inner_SG_weight, edge_num)
            if (density != 0 or Modularity != 0):
                fitness_SG = (float(density) + float(Modularity)) / 2 + 1 / (1 / float(density) +1 / float(Modularity))
            else:
                fitness_SG = 0
            subgraph_node.append(in_node)
        
            if(fitness_SG > raw_fitness_SG and actually_edges >= raw_expectation_edge):
                raw_fitness_SG = fitness_SG; raw_expectation_edge = actually_edges
                minNode = in_node

        if minNode:
            subgraph_node.remove(minNode)
            inner_node.remove(minNode)
        else:
            break        

    return subgraph_node, raw_fitness_SG, raw_expectation_edge

def modularity(graph, subgraph_node):
    G = graph; SG_neighbor_nodes = []; inner_SG_weight = 0; out_SG_weight = 0; edge_num = 0

    for value2 in subgraph_node:                                    #Neighbor nodes of subgraph
        SG_neighbor_nodes.extend(G.neighbors(value2))    
    SG_neighbor_nodes = list(set(SG_neighbor_nodes).difference(set(subgraph_node)))   

    if subgraph_node:                                   #Sum of Internal weight of subgraph
        for k1 in range(len(subgraph_node)):                   
            for k2 in range(k1+1, len(subgraph_node)):
                if G.has_edge(subgraph_node[k1], subgraph_node[k2]):
                    edge_weight = []; edge_weight = G.get_edge_data(subgraph_node[k1], subgraph_node[k2])
                    inner_SG_weight = float(inner_SG_weight) + float(edge_weight['weight'])
                    edge_num = edge_num + 1

    if SG_neighbor_nodes:                               #Sum of external weights of subgraph
        for k1 in SG_neighbor_nodes:
            for k2 in subgraph_node:
                if G.has_edge(k1,k2):
                    edge_weight = []; edge_weight = G.get_edge_data(k1, k2)
                    out_SG_weight = float(out_SG_weight) + float(edge_weight['weight'])
    
    Modularity = float(inner_SG_weight) / (float(inner_SG_weight) + float(out_SG_weight))  #Modularity value

    return inner_SG_weight, out_SG_weight, Modularity, edge_num

def weight_density(graph, subgraph_node, inner_SG_weight, edge_num):
    node_num = len(subgraph_node)
    if node_num == 1:
        density = 0
    else:
        density = float(inner_SG_weight) / ((float(node_num) * float(node_num - 1)) / 2)
    return density

def merge(complex_dict, fileName):
    del_num = []
    for key1 in complex_dict:
        for k in range(key1+1,len(complex_dict)):
            inter = list(set(complex_dict[key1][0]) & set(complex_dict[k][0]))
            if len(inter) == len(complex_dict[k][0]):
                del_num.append(k)
            if len(inter) == len(complex_dict[key1][0]) and len(inter) != len(complex_dict[k][0]):
                del_num.append(key1)
    
    del_num = set(del_num)
    for value in del_num:
        del complex_dict[value]
    
    f = open(fileName,"w")
    for key1 in complex_dict:
        for key2 in complex_dict[key1][0]:
            f.write(key2 + "\t")
        f.write("\n")
    f.close()

    return complex_dict

def calculate(G, k_shell_list, fileName): 
    complex_dict = defaultdict(list); k = 0
    for value1 in k_shell_list:
        if(int(G.degree(value1)) >= int(1)):
            subgraph_node = []
            subgraph_node.append(value1)
            fitness_SG = 0; oldfitness_SG = -1
            while fitness_SG > oldfitness_SG:
                oldfitness_SG = fitness_SG
                subgraph_node, fitness_SG, expectation_edge = inflate(G, subgraph_node)  #subgraph_node is a list
                subgraph_node, fitness_SG, expectation_edge = shrink(G, subgraph_node, fitness_SG, expectation_edge)

            if(fitness_SG > 0):
                complex_dict[k].append(subgraph_node)
                k += 1  
                f = open(fileName,"a+")
                for key1 in subgraph_node:
                    f.write(key1 + "\t")
                f.write("\n")
                f.close()

if __name__ == '__main__':
    
    time_start = time.monotonic()
    k_shell_dict = defaultdict(list); k_shell_num = []; 

    ###Please modify the file path
    filename = "/data/panyuliang/classification_methods/weight_graph/sum_weight.txt"  ###PPI network file
    outfile_first = "/data/panyuliang/cluster/cluster_result/sum_temporary_end.txt"   ###Temporary cache file
    outfile = "/data/panyuliang/cluster/cluster_result/sum_exlucde_duplicate_end.txt"    ###Complexes output file

    G = constructGraph(filename)
    print("G is constracted done...")
    k_shell_dict = k_shell(G)
    print("k_shell_dict is done...")

    ###You can set the number of nodes per thread
    threads = []
    for num in k_shell_dict.keys():
        k_shell_list = []
        k_shell_list.extend(k_shell_dict.get(num))      #Merge into one K_ shell 
        if int(num) >= 20:                              #K-value of node
            step = 1                                    #Each thread contains 1 node
            for i in range(0,len(k_shell_list),step):
                list_split_k_shell = []
                list_split_k_shell = k_shell_list[i:i+step]
                t = threading.Thread(target=calculate,args=(G, list_split_k_shell, outfile_first))
                threads.append(t)
        elif int(num) >=5 and int(num) <20: 
            step = 2
            for i in range(0,len(k_shell_list),step):
                list_split_k_shell = []
                list_split_k_shell = k_shell_list[i:i+step]
                t = threading.Thread(target=calculate,args=(G, list_split_k_shell, outfile_first))
                threads.append(t)
        else:
            t = threading.Thread(target=calculate,args=(G, k_shell_list, outfile_first))
            threads.append(t)            

    print(len(threads))
    print("threads start...")
    for key in threads:
        key.start()
    print("thread join...")
    for key in threads:
        key.join()
    print("ALL end")
    
    ####File deduplication
    end_complex = defaultdict(list); k = 0
    f = open(outfile_first,'r')
    for value in f:
        end_value = value.strip().split("\t")
        list_pc = []
        for key_value in end_value:
            list_pc.append(key_value)    
        end_complex[k].append(list_pc)
        k = k + 1
    end_complex = merge(end_complex, outfile)

    time_end = time.monotonic()
    print(f"span : {time_end-time_start:>9.2f}s")

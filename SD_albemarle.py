
import numpy as np
import networkx as nx
#import EoN
import matplotlib.pyplot as plt
import csv, random, pdb, sys
from IPython.core.debugger import set_trace
import collections
from itertools import combinations
import time
import multiprocessing
import threading
from multiprocessing import Manager, Pool
import copy

#each line: id1, id2
def read_graph(fname):
    G = nx.Graph()
    fp_reader = csv.reader(open(fname), delimiter = ',')
    count = 0
    for line in fp_reader:
        if line[1] != line[2]: 
            G.add_edge(line[1], line[2])
        count += 1
        if count >= 1000:
            break
    return G


#create components
#x: strategy vector where x[i] = 1 means i is vaccinated
def init_comp(G, x):
    
    # comp_id: {node u: component_id i}; mapping of each node to it's current component id
    # comp_len: {component_id i: length(int)}; mapping of component id to its length
    # comp_d: {component_id i: list of node in ith component}
    # max_comp_id: integer; each time we create a new component id so it will be helpful.
    comp_id = {}; comp_len = {}; comp_d = {}; max_comp_id = 0
    
    # first create a graph H which is a copy of G and then get connected component of H
    H = nx.Graph()
    for u in G.nodes(): 
        H.add_node(u)
        
    for e in G.edges():
        u = e[0]; v = e[1]
        if x[(u,v)] == 0 and x[(v,u)] == 0: #edge is not social distanced
            H.add_edge(u, v)
    comp = nx.connected_components(H)

    # comp is a list of list; for each component we assign a component id and 
    for c in list(comp):
        for u in c: 
            comp_id[u] = max_comp_id
        comp_len[max_comp_id] = len(list(c))
        comp_d[max_comp_id] = list(c)
        max_comp_id += 1
#     for u in x:
#         if x[u] == 1: comp_id[u] = -1
        
    return H, comp_d, comp_id, comp_len, max_comp_id


#remove edges from edge_list and split its comp
#use ids starting from comp_max_id + 1
def remove_edge(G, x, comp_d, comp_id, comp_len, comp_max_id, u, edge_list):
    
    C = set(comp_d[comp_id[u]])
    edge_list = set(edge_list)
    H = nx.Graph()
    for v in C: 
        H.add_node(v)

    # Remove edges that are in edge_list    
    for v1 in C: 
        for v2 in G.neighbors(v1):
            if v2 in C and (v1,v2) not in edge_list and (v2,v1) not in edge_list and x[(v1,v2)] != 1 and x[(v2,v1)] != 1: 
                H.add_edge(v1, v2)

    comp1 = nx.connected_components(H)
    comp = list(comp1).copy()

    for c in list(comp):
        comp_max_id += 1
        comp_d[comp_max_id] = list(c); 
        comp_len[comp_max_id] = len(c)
        for v in list(c): 
            comp_id[v] = comp_max_id
    return comp_d, comp_id, comp_len, comp_max_id

#add edges from edge_list and create comp
#use ids starting from comp_max_id + 1
def add_edge(G, x, comp_d, comp_id, comp_len, comp_max_id, u, edge_list):
    Tu = comp_d[comp_id[u]].copy()
    del comp_d[comp_id[u]]
    del comp_len[comp_id[u]]
    comp_max_id += 1
    S = set(Tu)

    for vprime in Tu: 
        comp_id[vprime] = comp_max_id
    
    for edge in edge_list:
        u1,v = edge
        T = []
        if comp_id[v] != comp_id[u] and comp_id[v] in comp_d:
            T = comp_d[comp_id[v]].copy()
        elif comp_id[v] == comp_id[u]:
            T = Tu
        S = set(S) | set(T)
        if comp_id[v] in comp_d: 
            del comp_d[comp_id[v]]
            del comp_len[comp_id[v]]
        for vprime in T: 
            comp_id[vprime] = comp_max_id

    #merge the components containing S into one
    comp_id[u] = comp_max_id
    comp_d[comp_max_id] = list(S)
    comp_len[comp_max_id] = len(S)

    return comp_d, comp_id, comp_len, comp_max_id


#flip strategy of node u
def update_strategy(x, G, H, comp_d, comp_id, comp_len, Csd, Cinf, comp_max_id, u, edge_list, split_flag):

    change = 0
    if split_flag == True:
        edge = edge_list[0]
        if x[edge] == 0 and x[(edge[1], edge[0])] != 1:
                # x[edge] 0-> 1
                comp_d, comp_id, comp_len, comp_max_id = remove_edge(G, x, comp_d, 
                                                                             comp_id, comp_len, comp_max_id, u, edge_list)
                change = 1
                for edge in edge_list:
                    x[edge] = 1
    
    elif split_flag == False: 
        edge = edge_list[0]
        if x[edge] == 1 and x[(edge[1], edge[0])] != 1:
            #x[edge] 1-> 0
            comp_d, comp_id, comp_len, comp_max_id = add_edge(G, x,
                                                              comp_d, comp_id, comp_len, comp_max_id, u, edge_list)
            change = 1
            for edge in edge_list:
                x[edge] = 0

    return x, comp_d, comp_id, comp_len, comp_max_id, change
    


def get_SD_components(G, x, comp_id, comp_len, comp_d, u):
    curr_comp = set(comp_d[comp_id[u]])
    H = nx.Graph()
    for v in curr_comp: 
        H.add_node(v)

    # Remove all incident edges from node u
    # Build a graph of nodes/edges only from curr_comp with no edges on u
    for v1 in curr_comp: 
        for v2 in G.neighbors(v1):
            if not (v1 == u or v2 == u) and (v2 in curr_comp) and x[(v1,v2)] != 1 and x[(v2,v1)] != 1: 
                H.add_edge(v1, v2)
                    
    comp1 = nx.connected_components(H)
    comp = list(comp1).copy()

    conn_comp_max_id = 0
    conn_comp_d = {}
    conn_comp_id = {}
    conn_comp_len = {}
    for c in list(comp):
        conn_comp_d[conn_comp_max_id] = list(c); 
        conn_comp_len[conn_comp_max_id] = len(c)
        for v in list(c): 
            conn_comp_id[v] = conn_comp_max_id
        conn_comp_max_id += 1

    conn_edge_group = collections.defaultdict(list)
    not_conn_edge_group = collections.defaultdict(list)

    for v in G.neighbors(u):
        if v in curr_comp and x[(u,v)] == 0 and x[(v,u)] == 0:
            neighbour_comp_id = conn_comp_id[v]
            conn_edge_group[neighbour_comp_id].append((u,v))
        elif v not in curr_comp and x[(u,v)] == 1:
            neighbour_comp_id = comp_id[v]
            not_conn_edge_group[neighbour_comp_id].append((u,v))
            
    not_conn_edge_group = list(not_conn_edge_group.values())
    conn_edge_group1 = []
    for key, val in conn_edge_group.items():
        conn_edge_group1.append((val, key))

    return conn_edge_group1, not_conn_edge_group, conn_comp_len

     
def print_analysis(comp_id, comp_len): 
    component_ids = np.unique(list(comp_id.values()))  
    component_lengths = [comp_len[i] for i in component_ids] 
    avg_comp_size = round(np.mean(component_lengths),2)
    max_comp_size = np.max(component_lengths)
    #print("Average component size: ", avg_comp_size)
    #print("Max component size: ", max_comp_size) 
    return avg_comp_size, max_comp_size   

#start at strategy x and run for T steps
def best_response(G, Csd, Cinf, x, T, epsilon=0.05):

    H, comp_d, comp_id, comp_len, comp_max_id = init_comp(G, x)
    
    start_time = time.time()
    for t in range(T):
        per_itr_st = time.time()
        change_count = 0
        for u in G.nodes():
            per_node_st = time.time()
            conn_edge_group, not_conn_edge_group, conn_comp_len = get_SD_components(G, x, comp_id, comp_len, comp_d, u)
            to_split = [0 for _ in range(len(conn_edge_group))]
            to_merge = [0 for _ in range(len(not_conn_edge_group))]
            
            for i, (conn_edge_list, component_id) in enumerate(conn_edge_group):
                social_dist_cost = 0
                for edge in conn_edge_list:
                    social_dist_cost += Csd[edge]
                
                if social_dist_cost < conn_comp_len[component_id]*Cinf[u]/(len(comp_id)+0.0):
                    to_split[i] = 1
                
            for i, not_conn_edge_list in enumerate(not_conn_edge_group):
                nbr_comp = {}
                social_dist_cost = 0
                for edge in not_conn_edge_list:
                    nbr_comp[comp_id[edge[1]]] = 1
                    social_dist_cost += Csd[edge]
                
                num_node = 0
                for j in nbr_comp:
                    num_node += comp_len[j] 
                
                if social_dist_cost > num_node*Cinf[u]/(len(comp_id)+0.0):
                    to_merge[i] = 1
            
            for i, (conn_edge_list, component_id) in enumerate(conn_edge_group):
                if to_split[i] == 1:
                    x, comp_d, comp_id, comp_len, comp_max_id, change = update_strategy(x, 
                                            G, H, comp_d, comp_id, comp_len, Csd, Cinf, comp_max_id, u, conn_edge_list, True)
                    change_count += change
                
            for i, not_conn_edge_list in enumerate(not_conn_edge_group):
                if to_merge[i] == 1:                    
                    x, comp_d, comp_id, comp_len, comp_max_id, change = update_strategy(x, 
                                            G, H, comp_d, comp_id, comp_len, Csd, Cinf, comp_max_id, u, not_conn_edge_list, False)
                    change_count += change
                    

        if change_count == 0:
            print("Total time in mins: ", round((time.time()-start_time)/60,4), "in", t+1, "iterations")
            avg_comp_size, max_comp_size = print_analysis(comp_id, comp_len)
            return x, change_count, avg_comp_size, max_comp_size, comp_id, comp_len, comp_d 
#    
    print("Total time in mins: ", round((time.time()-start_time)/60,4), "in", t+1, "iterations")
    avg_comp_size, max_comp_size = print_analysis(comp_id, comp_len)
    return x, change_count, avg_comp_size, max_comp_size, comp_id, comp_len, comp_d 

def save_degree_hist(comp_id, comp_len, comp_d, G, alpha):     
    component_ids = np.unique(list(comp_id.values()))  
    component_lengths = [comp_len[i] for i in component_ids] 
    index_max = np.argmax(component_lengths)
    max_comp = component_ids[index_max]
    max_comp_nodes = comp_d[max_comp]
    max_nodes_degrees = [G.degree(node) for node in max_comp_nodes]

    plt.hist(max_nodes_degrees, density=False, bins=20)  # `density=False` would make counts
    plt.ylabel('Num of nodes')
    plt.xlabel('Degree');
    plt.title("Histogram of degrees for Max Component Nodes; alpha: " + str(alpha) + "; len: " + str(len(max_comp_nodes)))
    plt.savefig("./out/fig/SD_BA_max_comp_nodes_alpha_" + str(alpha) + ".jpg")
    plt.close()

def save_graph_data(comp_id, comp_len, comp_d, G):     
    graph_data = {}
    graph_data['comp_id'] = comp_id
    graph_data['comp_len'] = comp_len
    graph_data['comp_d'] = comp_d
    graph_data['nodes'] = list(G.nodes())
    graph_data['edges'] = list(G.edges())
    graph_data['degree'] = dict(G.degree)
    return graph_data

def save_file(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            for item in row:
                f.write("%s\t" % item)
            f.write("\n")


def remove_topk_nodes(G, k):
    degree_tup = list(dict(G.degree).items())
    degree_tup = sorted(degree_tup, key=lambda x: -1*x[1])
    top_k_nodes = [x for x,d in degree_tup[:k]]
    G_res = copy.deepcopy(G)
    for node in top_k_nodes:
        G_res.remove_node(node)
    
    return G_res


def inital_comp_size(G):
    comp_id = {}; comp_len = {}; comp_d = {}; max_comp_id = 0
    H = nx.Graph()
    for u in G.nodes(): 
        H.add_node(u)
        
    for e in G.edges():
        u = e[0]; v = e[1]
        H.add_edge(u, v)
    comp = nx.connected_components(H)

    for c in list(comp):
        for u in c: 
            comp_id[u] = max_comp_id
        comp_len[max_comp_id] = len(list(c))
        comp_d[max_comp_id] = list(c)
        max_comp_id += 1
        
    return H, comp_d, comp_id, comp_len, max_comp_id



if __name__ == '__main__':
### run for a fixed network and fixed alpha
###########################################
    T = 100
    epsilon = 0.001
    alphavals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n = 100
    m = 2
    avg_file_name = './SD_albemarle_avg_v2.txt'
    raw_file_name = './SD_albemarle_raw_v2.txt' 
        
    raw_data = []
    num_times = 1
    np.random.seed(0);
    total_data = [[0 for _ in range(6)] for _ in range(len(alphavals))]

    total_data_with_k = []
    # G = read_graph('../household_graph_undirected.csv')
    G = nx.barabasi_albert_graph(100, 5)
    n = len(G.nodes())
    print("Original G num_edges:", len(G.edges()), " num_nodes:", len(G.nodes()))
    
    k_max_comp_dict = {}
    times = 1
    for k in range(99):
        k_node = int(k/100*len(G.nodes()))
        G_res = remove_topk_nodes(G, k_node )
        H, comp_d, comp_id, comp_len, comp_max_id = inital_comp_size(G_res)
        avg_comp_size, max_comp_size = print_analysis(comp_id, comp_len)
        #print("k: ", k, "avg_comp_size: ", avg_comp_size, "max_comp_size: ",  max_comp_size)
        k_max_comp_dict[k] = max_comp_size            
    
    print("Done k_max_comp_dict")
    
    total_graph_data = {}
    for ind, alpha in enumerate(alphavals):
        print("n/alpha: ", n/alpha)
        for k, max_comp_size in k_max_comp_dict.items():
            if max_comp_size < n/alpha:
                k_node = int(k/100*len(G.nodes()))
                G_res = remove_topk_nodes(G, k_node )
                break
                
        # n: number of nodes; m: Probability for edge creation; 
        # G = nx.fast_gnp_random_graph(n, m)
        print("Top k: ", k, "max_comp_size :", max_comp_size, "Num of nodes: ", len(G_res.nodes), "Num of edges: ", len(G_res.edges))
        x = {}; Csd = {}; Cinf = {};
        removed_nodes = set(G.nodes() - G_res.nodes())
        for edge in G.edges():
            # 0: no social distance; 1: social distance
            u,v = edge
            if u in removed_nodes:
                x[(u,v)] = 1
            else:
                x[(u,v)] = 0
            if v in removed_nodes:
                x[(v,u)] = 1
            else:
                x[(v,u)] = 0
            Csd[(u,v)] = 1;
            Csd[(v,u)] = 1;
            
        prev_sd_edges = len([i for i in x if x[i] == 1])
        for u in G.nodes():
            Cinf[u] = 1*float(alpha)            

        x, change_count, avg_comp_size, max_comp_size, comp_id, comp_len, comp_d = best_response(G, Csd, Cinf, x, T, epsilon)
        sd_edges = len([i for i in x if x[i] == 1])
        temp = [alpha, times, avg_comp_size, max_comp_size, sd_edges, prev_sd_edges, k]
        raw_data.append(temp)
        
        print("alpha: ", alpha, "change_count: ", change_count, "Social distanced edge: ", sd_edges, "prev_sd_edges: ", prev_sd_edges, "Len of x", len(x))
        print("avg_comp_size :", avg_comp_size, "max_comp_size :", max_comp_size, '\n')
        # total_data_with_k += total_data_with_var
        save_file(raw_file_name, raw_data)
        sys.stdout.flush()
        
        
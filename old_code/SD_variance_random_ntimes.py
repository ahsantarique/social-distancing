

'''
Csd[i] = cost of vaccination for node i
Cinf[i]: cost of infection for node i
x[i]: current strategy of node i
    x[i] = 1 ==> i is vaccinated
S(x): set of vaccinated nodes
comp(x): components formed by residual nodes
cost[i]: cost of node i

#Evaluating reduction in cost for node i
#return old cost - new cost
def reduction_in_cost(x, comp, cost, Csd, Cinf, i)
    if x[i] == 0, then return  cost[i] - Csd[i]
    if x[i] == 1
        A = {comp(j, x): j is a nbr of i }
        N = \sum_{X in A} |X|
        return  Csd[i] - N^2 Cinf[i]/n

#best response
def best_respose(Csd, Cinf)
    xinit: random strategy
    initialize comp, cost
    for t = 1.. T:
        for i in V:
            if reduction_in_cost(x, i) > 0:
                flip x[i]
                update comp

#possible efficiencies
    #uniform Csd, Cinf setting
        if there is a large comp X: check benefit of vaccination
        if all comp are small: check benefit of not vaccinating
'''


'''
Csd[i] = cost of vaccination for node i
Cinf[i]: cost of infection for node i
x[i]: current strategy of node i
    x[i] = 1 ==> i is vaccinated
S(x): set of vaccinated nodes
comp(x): components formed by residual nodes
cost[i]: cost of node i

#Evaluating reduction in cost for node i
#return old cost - new cost
def reduction_in_cost(x, comp, cost, Csd, Cinf, i)
    if x[i] == 0, then return  cost[i] - Csd[i]
    if x[i] == 1
        A = {comp(j, x): j is a nbr of i }
        N = \sum_{X in A} |X|
        return  Csd[i] - N^2 Cinf[i]/n

#best response
def best_respose(Csd, Cinf)
    xinit: random strategy
    initialize comp, cost
    for t = 1.. T:
        for i in V:
            if reduction_in_cost(x, i) > 0:
                flip x[i]
                update comp

#possible efficiencies
    #uniform Csd, Cinf setting
        if there is a large comp X: check benefit of vaccination
        if all comp are small: check benefit of not vaccinating
'''
import numpy as np
import networkx as nx
#import EoN
import matplotlib.pyplot as plt
import csv, random, pdb, sys
#from IPython.core.debugger import set_trace
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
        if count >= 100:
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
            if v2 in C and (v1,v2) not in edge_list and (v2,v1) not in edge_list: 
#                print(v1,v2)
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
        #comp_id[v] = comp_max_id
        for vprime in T: 
            comp_id[vprime] = comp_max_id
#             for vprime in x:
#                 if comp_id[vprime] not in comp_len: print('err3', vprime, u, v)
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
    # print(comp)
    # print(H.edges)
    # print(nx.draw(H))

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


def fn1(conn_edge_group, to_split, Csd, Cinf, conn_comp_len, comp_id):        
    for i, (conn_edge_list, component_id) in enumerate(conn_edge_group):
        social_dist_cost = 0
        for edge in conn_edge_list:
            social_dist_cost += Csd[edge]
        
        if social_dist_cost < conn_comp_len[component_id]*Cinf[u]/(len(comp_id)+0.0):
            to_split[i] = 1
    
def fn2(not_conn_edge_group, to_merge, Csd, Cinf, comp_len, comp_id):
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
     
def print_analysis(comp_id, comp_len): 
    component_ids = np.unique(list(comp_id.values()))  
    component_lengths = [comp_len[i] for i in component_ids] 
    avg_comp_size = round(np.mean(component_lengths),2)
    max_comp_size = np.max(component_lengths)
    print("Average component size: ", avg_comp_size)
    print("Max component size: ", max_comp_size) 
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
    
    print("Total time: ", (time.time()-start_time)/60)
    return x, change_count, avg_comp_size, max_comp_size, comp_id, comp_len, comp_d 

def save_degree_hist(comp_id, comp_len, comp_d, G, x, alpha):     
    component_ids = np.unique(list(comp_id.values()))  
    component_lengths = [comp_len[i] for i in component_ids] 
    index_max = np.argmax(component_lengths)
    max_comp = component_ids[index_max]
    max_comp_nodes = comp_d[max_comp]
    #max_nodes_degrees = [G.degree(node) for node in max_comp_nodes]
    max_nodes_degrees = []
    for u in max_comp_nodes:
        degree = 0
        for v in list(G.neighbors(u)):
            if x[(u,v)] != 1 and x[(v,u)] != 1:
                degree += 1
        if degree != 0:
            max_nodes_degrees.append(degree)
        
    all_nodes_degrees = list(dict(G.degree).values())
    plt.hist(all_nodes_degrees, density=False, bins=20)  # `density=False` would make counts
    plt.ylabel('Num of nodes')
    plt.xlabel('Degree');
    plt.title("Histogram of degrees for All Nodes; alpha: " + str(alpha) + "; len: " + str(len(G.nodes())))
    plt.savefig("./out/fig/all_nodes_alpha_" + str(alpha) + ".png")
    plt.close()
    #plt.show()
    plt.hist(max_nodes_degrees, density=False, bins=20)  # `density=False` would make counts
    plt.ylabel('Num of nodes')
    plt.xlabel('Degree');
    plt.title("Histogram of degrees for Max Component Nodes; alpha: " + str(alpha) + "; len: " + str(len(max_comp_nodes)))
    plt.savefig("./out/fig/SD_BA_max_comp_nodes_alpha_" + str(alpha) + ".png")
    plt.close()
    #plt.show()

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

if __name__ == '__main__':
### run for a fixed network and fixed alpha
###########################################
    T = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    alphavals = sys.argv[3].split(',')
    n = int(sys.argv[4]); 
    m = float(sys.argv[5])
    avg_file_name = sys.argv[6]
    raw_file_name = sys.argv[7]   
    
    raw_data = []
    num_times = 100
    np.random.seed(0);

    for times in range(num_times):
        ## random graphs
        # BA: n: Number of nodes; m: Number of edges to attach from a new node to existing nodes
        m = int(m)
        G = nx.barabasi_albert_graph(n, m)

        # n: number of nodes; m: Probability for edge creation; 
        # G = nx.fast_gnp_random_graph(n, m)
        print("Num of nodes: ", len(G.nodes), "Num of edges: ", len(G.edges))
        x = {}; Csd = {}; Cinf = {}; 
        total_graph_data = {}
        for edge in G.edges():
            # 0: no social distance; 1: social distance
            u,v = edge
            # x[(u,v)] = 0;
            # x[(v,u)] = 0
            random = np.random.randint(0, 4)
            if random >= 3:
                random1 = np.random.randint(0, 2);
                x[(u,v)] = random1
                x[(v,u)] = 1-random1; #np.random.randint(0, 2);
            else:
                x[(u,v)] = 0 #np.random.randint(0, 2);
                x[(v,u)] = 0;
            Csd[(u,v)] = 1;
            Csd[(v,u)] = 1;

        prev_sd_edges = len([i for i in x if x[i] == 1])
        for ind, alpha in enumerate(alphavals):
            print("alpha: ", alpha)
            for u in G.nodes():
                Cinf[u] = 1*float(alpha)
                
            x, change_count, avg_comp_size, max_comp_size, comp_id, comp_len, comp_d = best_response(G, Csd, Cinf, x, T, epsilon)
            sd_edges = len([i for i in x if x[i] == 1])
            print("num_times :", times)
            temp = [alpha, times, avg_comp_size, max_comp_size, sd_edges, prev_sd_edges]
            raw_data.append(temp)
            
            print("alpha: ", alpha, "change_count: ", change_count, "Social distanced edge: ", sd_edges, "prev_sd_edges: ", prev_sd_edges, "Len of x", len(x)//2, '\n')
            graph_data = save_graph_data(comp_id, comp_len, comp_d, G)
            save_degree_hist(comp_id, comp_len, comp_d, G, x, alpha)
            total_graph_data[alpha] = graph_data

        total_data = [[0 for _ in range(6)] for _ in range(len(alphavals))]
        for i in range(len(alphavals)):
            total_data[i][0] = alphavals[i]
            total_data[i][1] = times
            for j in range(2,6):
                val_list = [raw_data[k*len(alphavals)+i][j] for k in range(times)]
                std = round(np.std(val_list),1)
                mean = round(np.mean(val_list),3)
                total_data[i][j] = str(mean) + " \u00B1 " + str(std)

        save_file(raw_file_name, raw_data)
        save_file(avg_file_name, total_data)
        np.save('./out/SD_BA_total_graph_data.npy', total_graph_data)
        sys.stdout.flush()
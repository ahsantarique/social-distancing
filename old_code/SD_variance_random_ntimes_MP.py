

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
#     for u in x:
#         if x[u] == 1: comp_id[u] = -1
        
    return H, comp_d, comp_id, comp_len, max_comp_id


#remove edges from edge_list and split its comp
#use ids starting from comp_max_id + 1
def remove_edge(G, x, comp_d, comp_id, comp_len, comp_max_id, u, edge_list):
    
    C = set(comp_d[comp_id[u]])
    edge_list = set(edge_list)
    #print('ff', u, 'comp_d=', comp_d, 'comp_id=', comp_id, 'C=', C, 'comp_max_id=', comp_max_id)
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
    #print('fff', H.nodes(), H.edges(), list(comp))
    
    #print("C: ", C, "edge_list", edge_list)
    #print("comp: ", comp )
    
    # Changed ** (verify it)
    # comp_id = {}; comp_len = {}; comp_d = {};
    for c in list(comp):
        comp_max_id += 1
        comp_d[comp_max_id] = list(c); 
        comp_len[comp_max_id] = len(c)
        for v in list(c): 
            comp_id[v] = comp_max_id
    #print('gg', u,  'comp_d=', comp_d, 'comp_id=', comp_id, 'comp_max_id=', comp_max_id)    
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
    # print(conn_edge_group)
    # print(not_conn_edge_group)
    return conn_edge_group1, not_conn_edge_group, conn_comp_len


def fn1(conn_edge_group, to_split, Csd, Cinf, conn_comp_len, comp_id):        
    for i, (conn_edge_list, component_id) in enumerate(conn_edge_group):
        #print("conn_edge_list", conn_edge_list)
        #print("component_id", component_id, conn_comp_len[component_id])
        social_dist_cost = 0
        for edge in conn_edge_list:
            social_dist_cost += Csd[edge]
        
        #print("Conn cost: ", conn_edge_list, social_dist_cost, conn_comp_len[component_id]*Cinf[u]/(len(comp_id)+0.0))
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
        
        #print("Not conn cost: ", not_conn_edge_list, social_dist_cost, num_node*Cinf[u]/(len(comp_id)+0.0))
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
def best_response(ret_val, G, alphavals, ind, T, epsilon=0.05):
    alpha = alphavals[ind]

    x = {}; Csd = {}; Cinf = {}; #alpha = 10
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

    for u in G.nodes():
        Cinf[u] = 1*float(alpha)

    H, comp_d, comp_id, comp_len, comp_max_id = init_comp(G, x)
    
    start_time = time.time()
    for t in range(T):
        #u = random.choice(list(V));
        per_itr_st = time.time()
        change_count = 0
        for u in G.nodes():
            per_node_st = time.time()
            #print("u", u, "comp_id", comp_id)
            conn_edge_group, not_conn_edge_group, conn_comp_len = get_SD_components(G, x, comp_id, comp_len, comp_d, u)
            #print("conn_edge_group, not_conn_edge_group", conn_edge_group, not_conn_edge_group)
            
            to_split = [0 for _ in range(len(conn_edge_group))]
            to_merge = [0 for _ in range(len(not_conn_edge_group))]
            
            for i, (conn_edge_list, component_id) in enumerate(conn_edge_group):
                #print("conn_edge_list", conn_edge_list)
                #print("component_id", component_id, conn_comp_len[component_id])
                social_dist_cost = 0
                for edge in conn_edge_list:
#                    if x[edge] == 1 or x[(edge[1], edge[0])] == 1:
#                        print("RED FLAG CKP1")
#                        print(edge, conn_edge_list)
#                        prin
                    social_dist_cost += Csd[edge]
                
                #print("Conn cost: ", conn_edge_list, social_dist_cost, conn_comp_len[component_id]*Cinf[u]/(len(comp_id)+0.0))
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
                
                #print("Not conn cost: ", not_conn_edge_list, social_dist_cost, num_node*Cinf[u]/(len(comp_id)+0.0))
                if social_dist_cost > num_node*Cinf[u]/(len(comp_id)+0.0):
                    to_merge[i] = 1
            
            for i, (conn_edge_list, component_id) in enumerate(conn_edge_group):
                if to_split[i] == 1:
                    # split_flag = True if to_split[i]== 1
                    x, comp_d, comp_id, comp_len, comp_max_id, change = update_strategy(x, 
                                            G, H, comp_d, comp_id, comp_len, Csd, Cinf, comp_max_id, u, conn_edge_list, True)
                    change_count += change
                    #print("remove edge: ", conn_edge_list, comp_id)      
                
            for i, not_conn_edge_list in enumerate(not_conn_edge_group):
                if to_merge[i] == 1:
                    # split_flag = True if to_split[i]== 1
                    
                    x, comp_d, comp_id, comp_len, comp_max_id, change = update_strategy(x, 
                                            G, H, comp_d, comp_id, comp_len, Csd, Cinf, comp_max_id, u, not_conn_edge_list, False)
                    change_count += change
                    #print("add edge: ", not_conn_edge_list, comp_id)
                    
        if change_count == 0:
            print("Total time in mins: ", round((time.time()-start_time)/60,4), "in", t+1, "iterations for alpha: ", alpha)
            avg_comp_size, max_comp_size = print_analysis(comp_id, comp_len)
            ret_val[ind] = (x, change_count, avg_comp_size, max_comp_size)
            return ret_val
    
    print("Total time: ", (time.time()-start_time)/60)
    return x, change_count


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
    total_data = [[0 for _ in range(6)] for _ in range(len(alphavals))]

    for times in range(num_times):
        ## random graphs
        n = 100
        m = 2
        # n: Number of nodes; m: Number of edges to attach from a new node to existing nodes
        G = nx.barabasi_albert_graph(n, m)
        print("Num of nodes: ", len(G.nodes), "Num of edges: ", len(G.edges))

        prev_sd_edges = 45

        p_list = []
        manager = Manager()
        ret_val = manager.dict()
        for ind, alpha in enumerate(alphavals):
            print("alpha: ", alpha)

            p = multiprocessing.Process(target=best_response, args = (ret_val, G, alphavals, ind, T, epsilon))
            p.start()
            p_list.append(p)


        for ind, p in enumerate(p_list):
            p.join()
            x, change_count, avg_comp_size, max_comp_size = ret_val[ind]
            alpha = alphavals[ind]   
            #x, change_count, avg_comp_size, max_comp_size = best_response(G, Csd, Cinf, x, T, epsilon)
            sd_edges = len([i for i in x if x[i] == 1])
            print("num_times :", times)
            temp = [alpha, times, avg_comp_size, max_comp_size, sd_edges, prev_sd_edges]
            raw_data.append(temp)
            
            total_data_temp = [alpha, times+1, round((total_data[ind][2]*times + avg_comp_size)/(times+1),3), round((total_data[ind][3]*times + max_comp_size)/(times+1),3), 
                                round((total_data[ind][4]*times + sd_edges)/(times+1),3), round((total_data[ind][5]*times + sd_edges)/(times+1),3)]
            total_data[ind] = total_data_temp
            print("alpha: ", alpha, "change_count: ", change_count, "Social distanced edge: ", sd_edges, "prev_sd_edges: ", prev_sd_edges, "Len of x", len(x)//2, '\n')
        
        with open(raw_file_name, 'w') as f:
            for row in raw_data:
                for item in row:
                    f.write("%s\t" % item)
                f.write("\n")
        
        total_data_with_var = copy.deepcopy(total_data)
        for i in range(len(alphavals)):
            for j in range(2,6):
                val_list = [raw_data[k*len(alphavals)+i][j] for k in range(times)]
                std = round(np.std(val_list),1)
                mean = round(np.mean(val_list),3)
                total_data_with_var[i][j] = str(mean) + " \u00B1 " + str(std)

        with open(avg_file_name, 'w') as f:
            for row in total_data_with_var:
                for item in row:
                    f.write("%s\t" % item)
                f.write("\n")

        sys.stdout.flush()
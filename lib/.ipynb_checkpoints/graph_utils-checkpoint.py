import numpy as np
import networkx as nx
import EoN
import matplotlib.pyplot as plt
import csv, random, pdb, sys
from IPython.core.debugger import set_trace
import copy
import pickle as pk
import random

'''
Cvacc[i] = cost of vaccination for node i
Cinf[i]: cost of infection for node i
x[i]: current strategy of node i
    x[i] = 1 ==> i is vaccinated
S(x): set of vaccinated nodes
comp(x): components formed by residual nodes
cost[i]: cost of node i

#Evaluating reduction in cost for node i
#return old cost - new cost
def reduction_in_cost(x, comp, cost, Cvacc, Cinf, i)
    if x[i] == 0, then return  cost[i] - Cvacc[i]
    if x[i] == 1
        A = {comp(j, x): j is a nbr of i }
        N = \sum_{X in A} |X|
        return  Cvacc[i] - N^2 Cinf[i]/n

#best response
def best_respose(Cvacc, Cinf)
    xinit: random strategy
    initialize comp, cost
    for t = 1.. T:
        for i in V:
            if reduction_in_cost(x, i) > 0:
                flip x[i]
                update comp

#possible efficiencies
    #uniform Cvacc, Cinf setting
        if there is a large comp X: check benefit of vaccination
        if all comp are small: check benefit of not vaccinating
'''


#each line: id1, id2
def read_graph(fname):
    G = nx.Graph()
    fp_reader = csv.reader(open(fname), delimiter = ' ')
    headers = next(fp_reader) 
    count = 0
    for line in fp_reader:
        if line[1] != line[2]: 
            G.add_edge(line[1], line[2])
        count += 1
    return G

#create components
#x: strategy vector where x[i] = 1 means i is vaccinated
def init_comp(G, x):
    # comp_id: {node u: component_id i}; mapping of each node to it's current component id
    # comp_len: {component_id i: length(int)}; mapping of component id to its length
    # comp_d: {component_id i: list of node in ith component}
    # max_comp_id: integer; each time we create a new component id so it will be helpful.
    
    
    print("init comp")
    comp_id = {}; comp_len = {}; comp_d = {}; max_comp_id = 0
    
    
    d_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    K = 50
    nodes = [k for k, v in d_sorted]

    print("degree done")
    
    for i in nodes[:K]:
        x[i] = 1
    
    H = nx.Graph()
    for u in G.nodes(): 
        H.add_node(u)
    for e in G.edges():
        u = e[0]; v = e[1]
        if x[u] == 0 and x[v] == 0: #both nodes unvacccinated
            H.add_edge(u, v)
    comp = nx.connected_components(H)
    
    print("connected components done")
    
    for c in list(comp):
        max_comp_id += 1
        for u in c: 
            comp_id[u] = max_comp_id
        comp_len[max_comp_id] = len(list(c))
        comp_d[max_comp_id] = list(c)

    print("end init comp")
    
    return H, comp_d, comp_id, comp_len, max_comp_id

def comp_cost(G, x, Cvacc, Cinf, p):
    cost = {}
    #calculate #unvacc nbrs for each node
    H = nx.Graph()
    for e in G.edges():
        H.add_edge(e[0], e[1])
    for i in x:
        if x[i] == 1: 
            cost[i] = Cvacc[i]
        else: 
            cost[i] = (1 + p*(H.degree(i)))/(len(x)+0.0)
    return cost

def comp_cost2(x, comp_id, comp_len, Cvacc, Cinf):
    cost = {}
    for i in x:
        if x[i] == 1: 
            cost[i] = Cvacc[i]
        else: 
            cost[i] = comp_len[comp_id[i]]*Cinf[i]/(len(x)+0.0)
    return cost

#return reduction in cost if node u flips its strategy
def reduction_in_cost(G, x, p, cost, Cvacc, Cinf, u):
    if x[u] == 0: 
        return  cost[u] - Cvacc[u]
    if x[u] == 1:
        num_unvacc_nbrs = 0
        for v in G.neighbors(u): 
            if x[v] == 0: num_unvacc_nbrs += 1
        return cost[u] - (1 + p*num_unvacc_nbrs)/(len(x)+0.0)


def check_NE(G, x, p, cost, Cvacc, Cinf):
    num_violated = 0
    for u in G.nodes():
        if reduction_in_cost(G, x, p, cost, Cvacc, Cinf, u) > 0: num_violated += 1
    return num_violated
            
#flip strategy of node u
def update_strategy(G, x, p, cost, Cvacc, Cinf, u):

    if x[u] == 0:
        x[u] = 1
        cost[u] = Cvacc[u]
        return x, cost

    else: #x[u] = 1
        x[u] = 0
        num_unvacc_nbrs = 0
        for v in G.neighbors(u): 
            if x[v] == 0: num_unvacc_nbrs += 1
        cost[u] = (1 + p*num_unvacc_nbrs)/(len(x)+0.0)
        
        return x, cost
        
def print_analysis(comp_id, comp_len): 
    component_ids = np.unique(list(comp_id.values()))  
    component_lengths = [comp_len[i] for i in component_ids] 
    avg_comp_size = round(np.mean(component_lengths),2)
    max_comp_size = np.max(component_lengths)
    #print("Average component size: ", avg_comp_size)
    #print("Max component size: ", max_comp_size) 
    return avg_comp_size, max_comp_size  

#start at strategy x and run for T steps
def best_response(G, Cvacc, Cinf, x, T, p, epsilon=0.05):

    cost = comp_cost(G, x, Cvacc, Cinf, p)
    V = G.nodes(); itrn = 0
    
    for t in range(T):
        #u = random.choice(list(V)); 
        num_updated = 0
        for u in G.nodes():
#             itrn += 1
#             if (itrn % 10 == 0): print(itrn)
            if reduction_in_cost(G, x, p, cost, Cvacc, Cinf, u) > 0:
                x, cost = update_strategy(G, x, p, cost, Cvacc, Cinf, u)
                num_updated += 1

        if num_updated == 0: break
    
    print("Time to converge: ", t)
    return x, check_NE(G, x, p, cost, Cvacc, Cinf)



#start at strategy x and run for T steps
def best_response_v2(G, Cvacc, Cinf, x, T, p, epsilon=0.05):
    if len(x) == 0:
        for u in G.nodes(): x[u] = np.random.randint(0, 2)
    
    H, comp_d, comp_id, comp_len, comp_max_id = init_comp(G, x)
    #print('x', x)
    cost = comp_cost2(x, comp_id, comp_len, Cvacc, Cinf)
    V = G.nodes(); itrn = 0
    for t in range(T):
        #u = random.choice(list(V)); 
        num_updated = 0
        for u in G.nodes():
#             itrn += 1
#             if (itrn % 10 == 0): print(itrn)
            if reduction_in_cost(G, x, p, cost, Cvacc, Cinf, u) > 0:
                x, cost = update_strategy(G, x, p, cost, Cvacc, Cinf, u)
                num_updated += 1
        if num_updated <= epsilon*len(x): return x, num_updated
    
    print("Time to converege: ", t)
    
    return x, num_updated


def save_file(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            for item in row:
                f.write("%s\t" % item)
            f.write("\n")


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
    plt.savefig("../out/fig/max_comp_nodes_alpha_" + str(alpha) + ".png")
    plt.show()

def save_graph_data(comp_id, comp_len, comp_d, G):     
    graph_data = {}
    graph_data['comp_id'] = comp_id
    graph_data['comp_len'] = comp_len
    graph_data['comp_d'] = comp_d
    graph_data['nodes'] = list(G.nodes())
    graph_data['edges'] = list(G.edges())
    graph_data['degree'] = dict(G.degree)
    return graph_data



def remove_topk_nodes(G, k):
    degree_tup = list(dict(G.degree).items())
    degree_tup = sorted(degree_tup, key=lambda x: -1*x[1])
    top_k_nodes = [x for x,d in degree_tup[:k]]
    G_res = copy.deepcopy(G)
    for node in top_k_nodes:
        G_res.remove_node(node)
    
    return G_res

def exp_infsize(G, x, p):
    num_itrn = 100
    z = 0
    for i in range(num_itrn):
        H = nx.Graph()
        for u in G.nodes(): 
            if x[u] == 0: H.add_node(u)
        for e in G.edges():
            if x[e[0]] == 0 and x[e[1]] == 0 and random.random() <= p:
                H.add_edge(e[0], e[1])
        comp = nx.connected_components(H)
        z1 = 0
        for c in list(comp):
            a = len(list(c))
            z1 += a*a
        z += z1/(len(x)+0.0)
        
    return z/(num_itrn*len(x)+0.0)



def get_topk_nodes(G, k):
    degree_tup = list(dict(G.degree).items())
    degree_tup = sorted(degree_tup, key=lambda x: -1*x[1])
    top_k_nodes = [x for x,d in degree_tup[:k]]
    return top_k_nodes
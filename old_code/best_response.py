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
import numpy as np
import networkx as nx
#import EoN
import matplotlib.pyplot as plt
import csv, random, pdb, sys
#from IPython.core.debugger import set_trace

#each line: id1, id2
def read_graph(fname):
    G = nx.Graph()
    fp_reader = csv.reader(open(fname), delimiter = ',')
    next(fp_reader)
    for line in fp_reader:
        if line[1] != line[2]: G.add_edge(line[1], line[2])
#     for line in fp_reader:
#         G.add_edge(line[0], line[1])
    return G

#create components
#x: strategy vector where x[i] = 1 means i is vaccinated
def init_comp(G, x):
    
    comp_id = {}; comp_len = {}; comp_d = {}; max_comp_id = 0
    
    H = nx.Graph()
    for u in G.nodes(): H.add_node(u)
    for e in G.edges():
        u = e[0]; v = e[1]
        if x[u] == 0 and x[v] == 0: #both nodes unvacccinated
            H.add_edge(u, v)
    comp = nx.connected_components(H)
    
#     nx.set_node_attributes(H, 'state', [])
#     for u in G.nodes(): 
#         H.add_node(u); 
#         if u in x:
#             H.node[u]['state'] = x[u]
#         else:
#             print(u, 'err'); 
#             break
#     for e in G.edges():
#         u = e[0]; v = e[1]
#         if H.node[u]['state'] == 0 and H.node[v]['state'] == 0: #both nodes unvacccinated
#             H.add_edge(u, v)
#     comp = nx.connected_components(H)
    
    for c in list(comp):
        max_comp_id += 1
        for u in c: comp_id[u] = max_comp_id
        comp_len[max_comp_id] = len(list(c))
        comp_d[max_comp_id] = list(c)
#     for u in x:
#         if x[u] == 1: comp_id[u] = -1
        
    return H, comp_d, comp_id, comp_len, max_comp_id

def comp_cost(x, comp_id, comp_len, Cvacc, Cinf):
    cost = {}
    for i in x:
        if x[i] == 1: cost[i] = Cvacc[i]
        else: cost[i] = comp_len[comp_id[i]]*Cinf[i]/(len(x)+0.0)
    return cost

#return reduction in cost if node u flips its strategy
def reduction_in_cost(G, x, comp_id, comp_len, cost, Cvacc, Cinf, u):
    if x[u] == 0: 
        return  cost[u] - Cvacc[u]
    if x[u] == 1:
        nbr_comp = {}; z = 1
        for v in G.neighbors(u): 
            if x[v] == 0: nbr_comp[comp_id[v]] = 1
        for j in nbr_comp: z += comp_len[j]
        return cost[u] - z*Cinf[u]/(len(x)+0.0)

def check_NE(G, x, comp, comp_id, comp_len, cost, Cvacc, Cinf):
    num_violated = 0
    for u in G.nodes():
        if reduction_in_cost(G, x, comp_id, comp_len, cost, Cvacc, Cinf, u) > 0: 
            num_violated += 1
    return num_violated

#remove node u and split its comp
#use ids starting from comp_max_id + 1
def remove_node(G, x, comp_d, comp_id, comp_len, comp_max_id, u):
    
    C = set(comp_d[comp_id[u]])
    C.remove(u); 
    del comp_d[comp_id[u]]; del comp_len[comp_id[u]]
    comp_max_id += 1; comp_id[u] = comp_max_id; comp_d[comp_max_id] = [u]; comp_len[comp_max_id] = 1
    #print('ff', u, 'comp_d=', comp_d, 'comp_id=', comp_id, 'C=', C, 'comp_max_id=', comp_max_id)
    H = nx.Graph()
    for v in C: H.add_node(v)
    for v1 in C: 
        for v2 in G.neighbors(v1):
            if v2 in C: H.add_edge(v1, v2)
    comp1 = nx.connected_components(H)
    comp = list(comp1).copy()
    #print('fff', H.nodes(), H.edges(), list(comp))
    
    for c in list(comp):
        comp_max_id += 1
        comp_d[comp_max_id] = list(c); comp_len[comp_max_id] = len(c)
        for v in list(c): comp_id[v] = comp_max_id
    #print('gg', u,  'comp_d=', comp_d, 'comp_id=', comp_id, 'comp_max_id=', comp_max_id)    
    return comp_d, comp_id, comp_len, comp_max_id

#add node u and create comp
def add_node(G, x, comp_d, comp_id, comp_len, comp_max_id, u):
    Tu = comp_d[comp_id[u]].copy()
    del comp_d[comp_id[u]]
    del comp_len[comp_id[u]]
    
    comp_max_id += 1
    S = set([u])

    for v in G.neighbors(u):
        if x[v] == 0: #v is not vaccinated
            if comp_id[v] != comp_id[u] and comp_id[v] in comp_d:
                T = set(comp_d[comp_id[v]].copy())
            elif comp_id[v] == comp_id[u]:
                T = set(Tu)
            S = S | T
            if comp_id[v] in comp_d: 
                del comp_d[comp_id[v]]
                del comp_len[comp_id[v]]
            #comp_id[v] = comp_max_id
            for vprime in T: comp_id[vprime] = comp_max_id
#             for vprime in x:
#                 if comp_id[vprime] not in comp_len: print('err3', vprime, u, v)
    #merge the components containing S into one
    comp_id[u] = comp_max_id
    comp_d[comp_max_id] = list(S)
    comp_len[comp_max_id] = len(S)

    return comp_d, comp_id, comp_len, comp_max_id
            
#flip strategy of node u
def update_strategy(x, G, H, comp_d, comp_id, comp_len, cost, Cvacc, Cinf, comp_max_id, u):

    if x[u] == 0:
        x[u] = 1
        comp_d, comp_id, comp_len, comp_max_id = remove_node(G, x, comp_d, 
                                                             comp_id, comp_len, comp_max_id, u)
        cost[u] = Cvacc[u]
        return x, comp_d, comp_id, comp_len, cost, comp_max_id

    else: #x[u] = 1
        x[u] = 0
        comp_d, comp_id, comp_len, comp_max_id = add_node(G, x,
                                                          comp_d, comp_id, comp_len, comp_max_id, u)
        cost[u] = comp_len[comp_id[u]]*Cinf[u]/(len(x)+0.0)
        return x, comp_d, comp_id, comp_len, cost, comp_max_id
        
def print_analysis(comp_id, comp_len): 
    component_ids = np.unique(list(comp_id.values()))  
    component_lengths = [comp_len[i] for i in component_ids] 
    avg_comp_size = round(np.mean(component_lengths),2)
    max_comp_size = np.max(component_lengths)
    print("Average component size: ", avg_comp_size)
    print("Max component size: ", max_comp_size) 
    return avg_comp_size, max_comp_size  

#start at strategy x and run for T steps
def best_response(G, Cvacc, Cinf, x, T, epsilon=0.05):
    if len(x) == 0:
        for u in G.nodes(): x[u] = np.random.randint(0, 2)
    
    H, comp_d, comp_id, comp_len, comp_max_id = init_comp(G, x)
    #print('x', x)
    cost = comp_cost(x, comp_id, comp_len, Cvacc, Cinf)
    V = G.nodes(); itrn = 0
    for t in range(T):
        #u = random.choice(list(V)); 
        num_updated = 0
        for u in G.nodes():
#             itrn += 1
#             if (itrn % 10 == 0): print(itrn)
            if reduction_in_cost(G, x, comp_id, comp_len, cost, Cvacc, Cinf, u) > 0:
                x, comp_d, comp_id, comp_len, cost, comp_max_id = update_strategy(x, 
                                    G, H, comp_d, comp_id, comp_len, cost, Cvacc, Cinf, comp_max_id, u)
                num_updated += 1

#                 if check_NE(G, x, comp_d, comp_id, comp_len, cost, Cvacc, Cinf) < epsilon*len(x):
#                     return x, check_NE(G, x, comp_d, comp_id, comp_len, cost, Cvacc, Cinf)
        if num_updated == 0:
            avg_comp_size, max_comp_size = print_analysis(comp_id, comp_len)
            return x, check_NE(G, x, comp_d, comp_id, comp_len, cost, Cvacc, Cinf), avg_comp_size, max_comp_size
    
    avg_comp_size, max_comp_size = print_analysis(comp_id, comp_len)
    return x, check_NE(G, x, comp_d, comp_id, comp_len, cost, Cvacc, Cinf), avg_comp_size, max_comp_size


#start at strategy x and run for T steps
def best_response_v2(G, Cvacc, Cinf, x, T, epsilon=0.05):
    if len(x) == 0:
        for u in G.nodes(): x[u] = np.random.randint(0, 2)
    
    H, comp_d, comp_id, comp_len, comp_max_id = init_comp(G, x)
    #print('x', x)
    cost = comp_cost(x, comp_id, comp_len, Cvacc, Cinf)
    V = G.nodes(); itrn = 0
    for t in range(T):
        #u = random.choice(list(V)); 
        num_updated = 0
        for u in G.nodes():
#             itrn += 1
#             if (itrn % 10 == 0): print(itrn)
            num_updated += update_if_reduce(x, G, H, comp_d, comp_id, comp_len, 
                                            cost, Cvacc, Cinf, comp_max_id, u)
        if num_updated <= epsilon*len(x): return x, num_updated
    return x, num_updated

def save_file(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            for item in row:
                f.write("%s\t" % item)
            f.write("\n")

if __name__ == '__main__':
### run for a fixed network and fixed alpha
##########################################
    
    T = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    alphavals = sys.argv[3].split(',')
    n = int(sys.argv[4]); 
    m = int(sys.argv[5])
    avg_file_name = sys.argv[6]
    raw_file_name = sys.argv[7]   
    
    raw_data = []
    num_times = 100
    np.random.seed(0);

    #### read from a fixed graph
#     fname = sys.argv[4]
#     G = read_graph(fname)

    ## random graphs

    print("Num of nodes: ", n, "m: ", m)
    for times in range(num_times):
        G = nx.barabasi_albert_graph(n, m)
        for ind, alpha in enumerate(alphavals):
            x = {}; Cvacc = {}; Cinf = {}; #alpha = 10
            for u in G.nodes():
                x[u] = np.random.randint(0, 2)
                #print(u, x[u])
                Cvacc[u] = 1; Cinf[u] = Cvacc[u]*float(alpha)

            #T = 500
            x, nviol, avg_comp_size, max_comp_size = best_response(G, Cvacc, Cinf, x, T, epsilon)
            num_vacc_nodes = len([i for i in x if x[i] == 1])
            temp = [alpha, times, avg_comp_size, max_comp_size, num_vacc_nodes]
            raw_data.append(temp)

            print("alpha: ", alpha, "Percent violated: ", nviol/len(x), "Num of vaccinated nodes: ", num_vacc_nodes)

        total_data = [[0 for _ in range(5)] for _ in range(len(alphavals))]
        for i in range(len(alphavals)):
            total_data[i][0] = alphavals[i]
            total_data[i][1] = times
            for j in range(2,5):
                val_list = [raw_data[k*len(alphavals)+i][j] for k in range(times)]
                std = round(np.std(val_list),1)
                mean = round(np.mean(val_list),3)
                total_data[i][j] = str(mean) + " \u00B1 " + str(std)

        save_file(raw_file_name, raw_data)
        save_file(avg_file_name, total_data)
        #sys.stdout.flush()

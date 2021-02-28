import numpy as np
import networkx as nx
import EoN
import matplotlib.pyplot as plt
import csv, random, pdb, sys
from IPython.core.debugger import set_trace
import copy
import pickle as pk
import random
import collections
from util import *
from strategy_vector_manipulator import *

'''
Cvacc[i] = cost of vaccination for node i
Cinf[i]: cost of infection for node i
x[i]: current strategy of node i
    x[i] = 1 ==> i is vaccinated
S(x): set of vaccinated nodes
comp(x): components formed by residual nodes
cost[i]: cost of node i

'''
#each line: id1, id2
def read_graph(fname):
    G = nx.Graph()
    fp_reader = csv.reader(open(fname), delimiter = ' ')
    headers = next(fp_reader) 
    count = 0
    time = {}
    for line in fp_reader:
        u = line[1]
        v = line[2]
        t = int(line[3])/3600
        if  u != v: 
            G.add_edge(u, v)
            time[(u,v)] = t
            time[(v,u)] = t
        count += 1
    return G, time

#return reduction in cost if node u flips its strategy
def reduction_in_cost(G, x, p, cost, Cvacc, Cinf, u):
    if x[u] == 0: 
        return  cost[u] - Cvacc[u]
    if x[u] == 1:
        num_unvacc_nbrs = 0
        for v in G.neighbors(u): 
            if x[v] == 0: num_unvacc_nbrs += p[(u,v)]
        return cost[u] - (1 + num_unvacc_nbrs)/(len(x)+0.0)



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
            if x[v] == 0: num_unvacc_nbrs += p[(u,v)]
        cost[u] = (1 + num_unvacc_nbrs)/(len(x)+0.0)
        
        return x, cost


def exp_infsize(G, x, p):
    num_itrn = 100
    z = 0
    for i in range(num_itrn):
        H = nx.Graph()
        for u in G.nodes(): 
            if x[u] == 0: H.add_node(u)
        for e in G.edges():
            u = e[0]
            v = e[1]
            if x[u] == 0 and x[v] == 0 and random.random() <= p[(u,v)]:
                H.add_edge(u, v)
        comp = nx.connected_components(H)
        z1 = 0
        for c in list(comp):
            a = len(list(c))
            z1 += a*a
        z += z1/(len(x)+0.0)
        
    return z/(num_itrn*len(x)+0.0)



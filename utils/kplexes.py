# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:13:57 2021

@author: Leyang Xue

"""
import os 
from utils import prunconnected 
from utils import noprun
from snap import *
import networkx as nx
import math
from collections import defaultdict
import time
import itertools
import uuid
from random import shuffle
from multiprocessing import Pool
from datetime import datetime

def Block(H,K):
    #find neighbours of nodes set in H
    
    S = set(K)
    for u in K: #Overl Cliques
        S.update(H.neighbors(u))

    return S

def clique_number(Allcliques):
    #assign each node a maximum clique value
    Cl = defaultdict(int)
    for K in Allcliques:
        for u in K:
            Cl[u] = max(Cl[u], len(K))
    return Cl

def Prune(G,k,m,Allcliques):
    
    Coreness = nx.core_number(G) #ks
    Cliqueness = clique_number(Allcliques) #assigned each node a clique size
    surv = [u for u in G.nodes() if Coreness[u] >= m-k and Cliqueness[u] >= math.floor(m/k)] #surviving node according to core and clique criterio
    H = G.subgraph(surv).copy() #Prune(G,k,m)
    
    return H

def Filter_edges(H,k,m,Allcliques):
    
    J  = nx.Graph() #clique graph   
    for K in Allcliques:
        if len(K) >= math.floor(m/k):
            #if size of clique > core criterion, save the clique and add it to new graph 
            J.add_edges_from((u,v) for (u,v) in itertools.combinations(K,2))
    
    cond = [(u,v) for (u,v) in H.edges() if not J.has_edge(u,v)] #remove some edges that not appear in the clique graph
    #print("Filter_edges process: %s remove %.3f percentage edges" % (datetime.today().strftime('%Y-%m-%d %H:%M:%S'), len(cond)/len(H.edges)))
    H.remove_edges_from(cond) #FILTER_EDGES(H,k,m)
    
    return H

def algorithm_statistic(start_time,end_time,G,H,algorithm):  
    
    run_time = end_time -start_time
    s = (len(G.nodes())-len(H.nodes()))/(len(G.nodes()))
    print(algorithm + ": percentage of reductional node %.3f, run time %.3f " %(s, run_time))
    
def Filter(G,k,m):
    
    Allcliques = list(nx.find_cliques(G))
    H = Prune(G,k,m,Allcliques) #filtering the nodes (1)coreness (2)cliqueness
    H_f = Filter_edges(H,k,m,Allcliques) #filtering those edges
    
    surv = set() #updating the node set with block
    for K in nx.find_cliques(H_f):
        
        #find block 
        B = Block(H_f,K) 
        HB = H_f.subgraph(B)
        
        #filtering the HB
        H_dot = Prune(HB,k,m,nx.find_cliques(HB))
        surv.update(H_dot.nodes())                                
    
    H = G.subgraph(surv)

    return H

def InitNeighbors(G):
    #return nodes neighbor set 
    
    neighbors_dic={} 
    for n in G.Nodes():
        neighbors_dic[n.GetId()] = set(n.GetOutEdges())
    
    return neighbors_dic

def load_kplex(file_name):
   
   f=open(file_name,'r+')
   kplex = []
   
   for each in f:
       
       s = each.strip('\n')
       plex = []
       for i in s.split(','):
           plex.append(int(i))
       kplex.append(plex)
   
   return kplex

def find_kplex(output,G1,k,mode,num_of_kplex,set_type="connected"):

    neighbors_dic = InitNeighbors(G1) 
    if(set_type=="connected" or set_type=="all"):
        
        #k-plex struceture is connected 
        if mode == "EnumIncExc": #if k>1, EnumIncEx could exacute
            file_name = "%s.%s" % (output,"connected_EnumIncExc")
            prunconnected.k = k
            result = prunconnected.run(G1,num_of_kplex, neighbors_dic, file_name)
            kplex = load_kplex(file_name)
            os.remove(file_name)
            
        if mode == "Enum": # if k>=1, Enum could exacute
            file_name = "%s.%s" % (output,"connected_Enum")
            noprun.k = k
            result = noprun.run(G1,num_of_kplex, neighbors_dic,file_name)        
            kplex = load_kplex(file_name)
            os.remove(file_name)
            
    #result is the time to run the enumerate the k-plex in the G1
    
    return kplex, result

def BuildGraphFromFile(file_name):
    
    import re
    
    G1 = TUNGraph.New() #generate the G with snap 
    stack =set() #nodes set 
    lines = [line.strip() for line in open(file_name)]
    
    for line in lines: #create G
        l = re.split('\t| ',line) #split the '16\t18' ['16','18'] 
        if(int(float(l[0])) not in stack):
           G1.AddNode(int(float(l[0])))
           stack.add(int(float(l[0])))
        if( int(float(l[1])) not in stack):
           G1.AddNode(int(float(l[1])))
           stack.add(int(float(l[1])))
        G1.AddEdge(int(float(l[0])),int(float(l[1])))
        G1.AddEdge(int(float(l[1])),int(float(l[0])))
        
    return G1

def runFromFile(filein,fileout,k,mode,num_of_kplex,set_type="connected"):
    
    G1 = BuildGraphFromFile(filein) #file format: a   b (node1 node2)
    [kplex,result]= find_kplex(fileout,G1,k,mode,num_of_kplex,set_type="connected")
    #print('run time of enumerating k-plex in G:',result)
    os.remove(filein) #remove the filein
    
    return kplex 

def mkdir(path):
    
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('--create new folder--:'+str(path))
    else:
        print('There is this folder!')
        
def all_plexes(G,k,file_address): #state-of-the-art exhaustive enum
    
    workdir   = os.path.join(file_address,"working_dir") #need to create the current dir
    berin     = os.path.join(workdir, "block.edgelist")
    berout    = os.path.join(workdir, "out")
    
    mkdir(workdir)#create new workdir
    #save the G so as to generate the G1 with specific type 
    #file_add is a file address used to save the network and save the result of k-plex 
    unique_filename = uuid.uuid4().hex
    filein  = "%s.%s" % (berin, unique_filename) #name of write in 
    fileout = "%s.%s" % (berout, unique_filename) #name of read out 
        
    nx.write_edgelist(G,filein)
    mode =  "EnumIncExc"
    num_of_kplex = 99999999 
    k_plexes = runFromFile(filein,fileout,k,mode,num_of_kplex,set_type="connected")
    
    for kplex in k_plexes: #return the kplex 
        yield kplex
        

def complete(G, clique, candidates):#reagein thinking 
    
    S = set(clique)
    for u in sorted(candidates):
        if u in S:
            continue
        compatible = True
        for v in S:
            if not G.has_edge(u,v):
                compatible = False
                break
        if compatible:
            #print(u, "OK")
            S.add(u)
        #else:
        #    print(u, "KO")
        
    return S

def is_parent_clique(G, target_nodes, candidate_parent):#reagein thinking 
    
    #G: H, target_nodes: C, candidate_parent: K
    
    parent_aux = complete(G, [min(target_nodes)], target_nodes)
    parent = complete(G, parent_aux, G.nodes())
    
    return parent == set(candidate_parent)

def process_clique_batch(args):
    
    H, k, m, cliques,batch_id,file_address = args
    print('%d process begins to find the kplex:',batch_id)
    
    results = []
    
    for K in cliques:
        B = Block(H,K) 
        HB = H.subgraph(B)
        
        for P in all_plexes(HB, k,file_address):
            if len(P) >= m and is_parent_clique(H, P, K):
                results.append(P)
                
    
    return results
        
    
def large_kplex_process(G, k,m,file_address,n=None):
    '''
    This program could be run with parallel computing, it mainly find the larger kplexes
    when n equals to the number of all cliques in the G, it means that find the maximum k-plexes in the G

    Parameters
    ----------
    G : graph
        network.
    k : int
        k value, it determines the structure of subgraph, when the k is larger, 
        the density of subgraph is sparse. Generally, the k is set to 2. 
    m : int
        the size of k-plex,means the number of nodes in the k-plex.
    n : int
        the finding space of k-plex, if the n is larger, it means that 
        there is large probability to find the larger kplex, but it computational time is high.
    file_address : str
        file address.

    Yields
    ------
    kplexs : list
        identified kplex structure.
    '''
    start_time = time.time()
    procnum = 30
    #threaddict = {}
    
    H = Filter(G,k,m) 
    all_cliques = list(nx.find_cliques(H))
    shuffle(all_cliques)
    
    #write the program by leyangx
    if n == None:
        number_of_clique = len(all_cliques)
    else:
        number_of_clique = min(n,len(all_cliques))
    
    print('the number of all cliques in the filted network:', len(all_cliques))
    cliques = all_cliques[:number_of_clique]
    print('the number of cliques in the greedy algorithms:', len(cliques))

    batch_size = math.ceil(len(cliques)/procnum) #the number of task exacued by each procnum 
    print('the number of task to find the kplex in each process:',batch_size)
    args = []
    if batch_size > 0:
        for (i,l) in enumerate(range(0,len(cliques),batch_size)):
            clique_seq = cliques[l:l+batch_size]
            args.append([H,k,m,clique_seq,i,file_address])
               
    pool = Pool(processes=procnum)    
    results = pool.map(process_clique_batch,args)
    kplexs = []
    for kplex in results:
        kplexs.append(kplex)
    
    return kplexs    

    end_time = time.time()
    algorithm_statistic(start_time,end_time,G,H,'large_kplex_process')  
    
    #number_of_clique = min(n,len(all_cliques)
    #cliques = all_cliques[:number_of_clique]
    #batch_size = math.ceil(len(cliques)/procnum) #the number of task exacued by each procnum 
    #manager    = multiprocessing.Manager()
    #resultdic  = manager.dict()
    #if (batch_size > 0):
    #    for (i,l) in enumerate(range(0, len(cliques), batch_size)):
    #        threaddict[i] = multiprocessing.Process(target=process_clique_batch, args=(H, k, m, cliques, l, l + batch_size, i, resultdic,file_address), daemon=True)
    #        threaddict[i].start()    
    #    print("%s started %d threads (batch_size=%d, procnum=%d)" % (datetime.today().strftime('%Y-%m-%d %H:%M:%S'), i+1, batch_size, procnum))        
    #for (i,p) in threaddict.items():
        #print('i',i)
        #print('p',p)
    #    p.join()        
    #    print("%s ended thread %d" % (datetime.today().strftime('%Y-%m-%d %H:%M:%S'), i))        
    #    yield from resultdic[i]

def large_kplex(G,k,m,file_address):
    
    start_time = time.time()
    
    #filtering, computing a sub-graph of G according Prune strategy
    H = Filter(G,k,m) 
    cliques = list(nx.find_cliques(H)) 
    shuffle(cliques)

    #number_of_clique = min(n,len(all_cliques))
    #cliques = all_cliques[:number_of_clique]
    print('the number of cliques:', len(cliques))
    
    for i,K in enumerate(cliques):
        
        #print('%d enumeration time:'% i)
        #generating the Block of cliques
        B = Block(H,K) 
        HB = H.subgraph(B)
        
        #find all plexes
        for C in all_plexes(HB, k,file_address):
            if len(C) >= m and is_parent_clique(H, C, K):
                yield C
    
    end_time = time.time()
    algorithm_statistic(start_time,end_time,G,H,'large_kplex')  
        
def max_plexes_greedy(G, k,file_address):
    
    o = max(len(K) for K in nx.find_cliques(G))
    m = o
    print("[greedy] try m:", m)
    Plist = list(large_kplex(G,k,m,file_address)) #reuse clique computation if needed
    maxsz = max(len(P) for P in Plist)
    results =[]
    for P in Plist:
        if len(P) == maxsz:
            results.append(P)
    return results

def max_plexes_greedy_process(G, k,file_address,n):
    
    o = max(len(K) for K in nx.find_cliques(G))
    m = o
    print("[greedy] try m:", m)
    
    Plist = large_kplex_process(G,k,m,file_address,n) #reuse clique computation if needed
    # while len(Plist) < 1:
    #     print('iteration....')
    #     Plist = list(large_kplex_process(G,k,m,n,file_address)) #reuse clique computation if needed
    maxsz = max(len(P) for P in Plist)
    results =[]
    for P in Plist:
        if len(P) == maxsz:
            results.extend(P)
            
    return results
# =============================================================================
#             
# if __name__ == '__main__':
#     
#     os.getcwd()
#     
#     data = np.loadtxt('ca-netscience.mtx')
#     G = nr.load_network(data)
#     
#     k=2
#     m=4
#     start_time = time.time()
#     kplex_list= list(large_kplex(G,k,m))
#     end_time = time.time()
#     print("Maximum k-plex for singal threads: run time %.3f " %(end_time-start_time))
#     
#     start_time = time.time()
#     kplex_process = list(large_kplex_process(G,k,m))
#     end_time = time.time()
#     print("Maximum k-plex for multiple threads: run time %.3f " %(end_time-start_time))
#     
# =============================================================================

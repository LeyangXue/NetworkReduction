# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:09:14 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors
import matplotlib.patches as pc
import pandas as pd
import seaborn as sns
import pickle as pk
import itertools
import random
import os

def SaveIM(data,path,filename):
    '''
    Save the data to a directory of IM
    
    Parameters
    ----------
    data : any format
        data to save .
    path : str
        path to save the data.
    filename : str
        file names.

    Returns
    -------
    None.

    '''
    pk.dump(data,open(path+'/IM/'+filename,'wb'))

def SaveER(data,path,filename):
    '''
    Save the data to a directory of ER
    
    Parameters
    ----------
    data : any format
        data to save .
    path : str
        path to save the data.
    filename : str
        file names.

    Returns
    -------
    None.

    '''
    pk.dump(data,open(path+'/ER/'+filename,'wb'))

def SaveSS(data,path,filename):
    '''
    Save the data to a directory of SS
    
    Parameters
    ----------
    data : any format
        data to save .
    path : str
        path to save the data.
    filename : str
        file names.

    Returns
    -------
    None.

    '''
    pk.dump(data,open(path+'/SS/'+filename,'wb'))
    
def SaveSSspread(data,path,filename):
    '''
    Save the data to a directory of SS/NR
    
    Parameters
    ----------
    data : any format
        data to save .
    path : str
        path to save the data.
    filename : str
        file names.

    Returns
    -------
    None.

    '''
    pk.dump(data,open(path+'/SS/spread/'+filename,'wb'))
    
def findBlockSeed(G):
    
    maxk = 0
    degree = dict(nx.degree(G))
    
    for node in degree.keys():
        if degree[node] >  maxk:
           maxk = degree[node]
           maxkNode = node
           
    return maxkNode

def identifyTopN(centrality,N):
    '''
    
    identify the top-n nodes for a given centrality
    
    Parameters
    ----------
    centrality : dict
        network centrality.
    N : int
        the number of nodes.

    Returns
    -------
    maxNode : array
       identified nodes.

    '''
    nodeweight= np.array([[node,centrality[node]] for node in centrality.keys()])
    maxindex = np.argsort(-nodeweight[:,1])[0:N]
    maxNode = nodeweight[maxindex,0]  
    
    return maxNode

def LocateSeed(G,ReducedNodes,topN):
    '''
    For given top-n supernodes, identifying the seed by selecting one node in each supernode

    Parameters
    ----------
    G : graph
        network.
    ReducedNodes : dict
        map relation.
    topN : array
        top-n supernode.

    Returns
    -------
    seed : list
        identified seed nodes.
    '''
    
    block = {}
    seed = []
    for each in topN:
       if each in ReducedNodes.keys():
          block[each]=cg.TrackBackNodes([each],ReducedNodes) 
       else:
          seed.append(each)
          
    for newnodes in block.keys():
        block_nodes = block[newnodes]
        blockG = nx.subgraph(G, block_nodes)
        seed.append(findBlockSeed(blockG))
        
    return seed

def Identify_seed(path,G,centrality,n):
    '''
    identify top-n supernodes according to the centrality  
    
    Parameters
    ----------
    path : str
        The path to load the sizeCliques and cliquelabels.
    G : graph
        network.
    centrality : dict
        network centrality.
    n : int
        the number of seeds.

    Returns
    -------
    seed : list
        identified seed nodes.
    '''
    #load the map relation
    ReducedNodes = cg.ReducedNodeMap(path)
    
    #identify top-n nodes according to the centrality 
    topn = identifyTopN(centrality,n) 
    
    #For given top-n supernodes, determine the seed by selecting one node in each supernode
    seed = LocateSeed(G,ReducedNodes,topn) 
    
    return seed

def adaptCentParal(args):
    '''
    

    Parameters
    ----------
    args : list
        parameter sets.
    G : network
    N : the number of seeds 
    func_name : centrality name
    func: centrality function
        
    Returns
    -------
    func_name : str
        centrality name.
    cseeds : dict
        initial seeds.

    '''
    G,N,func_name,func = args
    print('centrality:',func_name)
    
    #copy the network and create the array and dict to save the result
    subG = copy.deepcopy(G) 
    cseeds = {}
    seeds= []
    
    #calculate the adpative centrality 
    for i in np.arange(1,N):
        if func_name in ['Eigen.','Katz']:
            central = func(subG,max_iter=100000)
        else:
            central = func(subG)
        node = identifyTopN(central,1)
        seeds.append(node[0]) 
        cseeds[i]=copy.deepcopy(seeds)
        subG.remove_node(node[0])
            
    return func_name,cseeds

def adaptNodeCent(networkpath,resultpath,G,netIndex,clique,N):
    '''
    identify the initial seeds for different adaptive centraility 

    Parameters
    ----------
    networkpath : str
        path to load the network
    resultpath : str
        path to save the result.
    G : network
        graph.
    netIndex : dict
        map relation.
    clique : int
        value of k-clique, it is used to identify the seed node by network reduction method on basic of k-clique.
    N : int
        the number of seeds.

    Returns
    -------
    N_cseeds : dict
        seed nodes for different adaption centrality.

    '''
    C_nseeds ={} 
    NR5 = cg.load(networkpath+'/'+str(netIndex[clique])+'_Wnewnodes')
    
    #calculate the centrality of network reduction: identify the important nodes according to the weight on 5-clique CGNS
    NR5seeds={}
    for n in np.arange(1,N):
        NR5seeds[n] = Identify_seed(networkpath,G,NR5,n)
    C_nseeds['NR5'] = NR5seeds
    
    #save the function of centrality 
    cfunction={}
    cfunction['k'] = nx.degree_centrality
    cfunction['K-core'] = nx.core_number
    cfunction['Betwn.'] = nx.betweenness_centrality
    cfunction['Closn.'] = nx.closeness_centrality
    cfunction['Eigen.'] = nx.eigenvector_centrality
    cfunction['Katz'] = nx.katz_centrality
    cfunction['Subgh.'] = nx.subgraph_centrality
    cfunction['CI'] = cg.collective_influence
    cfunction['NB'] = cg.nb_centrality
    
    #calculate the commom network centrality
    for func_name in cfunction.keys():
       args = [G,N,func_name,cfunction[func_name]]
       [func_name,cseeds] = adaptCentParal(args)
       SaveIM(cseeds,resultpath,func_name +' adaption centrality')
       C_nseeds[func_name] = cseeds
       
    #transform the keys of centrality by the number of seeds
    N_cseeds={}
    for n in np.arange(1,N):
        cseeds = {}
        for centra_name in C_nseeds.keys():
            nseeds = C_nseeds[centra_name]
            cseeds[centra_name]= nseeds[n]
        N_cseeds[n] = cseeds
   
    SaveIM(N_cseeds,resultpath,'N_cseeds')
     
    return N_cseeds

def transform_Crhos(nspread,betas):
    '''
    transform the format of datasets:  from the dict to matrix

    Parameters
    ----------
    nspread : dict
        spread datasets.
    betas : list
        infection rate.

    Returns
    -------
    betarhos_matrix : dict
        spread datasets.
    cindex : dict
        map realtion between centrality and index.

    '''
    betarhos_matrix = {}
    for beta in betas:
        betaindex = int(round(beta*100,0))
        crhos_matrix = np.zeros((51,10))
        for n in nspread.keys():
            crhos = nspread[n]
            cindex = {c:i for i, c in enumerate(crhos.keys())}
            for c in crhos.keys():
                crhos_matrix[n,cindex[c]] = crhos[c][betaindex]
        betarhos_matrix[beta] = crhos_matrix
    
    return betarhos_matrix,cindex

def transform_CrhosBeta(G,nspread,n,cindex,resultpath):
    '''
    For a given n seeds, normalizing the rho by rho(centrality)/max_rho(centrality)
    
    Parameters
    ----------
    G : graph
        network.
    nspread : dict
        spread datasets.
    n : int
        the number of initial seeds.
    cindex : dict
        map relation between method and index.
    resultpath : str
        path to save the results.

    Returns
    -------
    normMatrix : array
        normalized performance for different centrality, x \in [0,1].
    methodindex : array
        the best method to achieve the influence maximization.

    '''
    nrhos = nspread[n]
    centr_matrix = np.zeros((10,31))
    for centr in nrhos.keys():
        centr_matrix[cindex[centr],:] = nrhos[centr]/G.order()
    
    centr_matrix = centr_matrix.T 
    
    #normalize the element by rho(centrality)/max_rho(centrality)
    methodindex = np.argmax(centr_matrix,axis=1)
    normMatrix = np.zeros(centr_matrix.shape)
    for i in np.arange(centr_matrix.shape[0]):
        arginx = methodindex[i]
        normMatrix[i,:] = centr_matrix[i,:]/centr_matrix[i,arginx]
    
    pk.dump(normMatrix,open(resultpath+'/IM/normMatrix','wb'))
    pk.dump(methodindex,open(resultpath+'/IM/methodindex','wb'))
    
    return normMatrix,methodindex

def PathLength(G,seed):
    '''
    calculate the average length of shortest path between any pairs of seeds  

    Parameters
    ----------
    G : graph
        network.
    seed : list
        initial seeds.

    Returns
    -------
    average_path : float
        the average length of shortest path
    '''
    path_length = 0
    pairs = list(itertools.combinations(seed, 2))
    for s in pairs:
        path_length += len(nx.shortest_path(G, s[0],s[1]))
    
    average_path = path_length/len(pairs)
    
    return average_path

def NR_seed(networkpath,resultpath,G,netIndex):
    '''
    identify the seed node by using the network reduction method
    
    Parameters
    ----------
    networkpath : str
        path to load the network and cliquelabels.
    resultpath : str
        path to load the result
    G : graph
        network.
    netIndex : dict
        map relation between k-clique CGNs and index.

    Returns
    -------
    nseed : dict
        identified seeds based on different k-clique CGNs.
    npath : dict
        lenghth of shortest path between any pairs of identified seeds.

    '''
    N = [10,20,30]
    nseed = {}
    npath = {}
    for n in N: 
        #for different number of seeds
        rseed = {}
        avg_path_length={}
        
        for clique in sorted(netIndex.keys())[:8]:
            #for different k-clique    
            #load the weight of nodes in k-clique CGNs
            centrality = cg.load(networkpath+'/'+str(netIndex[clique])+'_Wnewnodes')
            #identify the seed nodes according to the node weight in k-clique CGNs
            seed = Identify_seed(networkpath,G,centrality,n) 
            #store the seed 
            rseed[clique] = seed
            # the average length of shortest path between any pairs of seed
            avg_path_length[clique] = PathLength(G,seed)
        
        #store the seed and path length
        nseed[n] = rseed
        npath[n] = avg_path_length
    
    SaveIM(nseed,resultpath,'NR5_nseed')
    SaveIM(npath,resultpath,'NR5_npath')
    
    return nseed,npath

def Spreadmseed(G,alphas,mu,seeds,simulation,path,filename):
    '''
    Perform the numerical simulation over multiple seeds  

    Parameters
    ----------
    G : graph
        network.
    alphas : array
        infection rate.
    mu : float
        recovery probability.
    seeds : list
        initial seeds.
    simulation : int
        simulation times.
    path : str
        path to save the result.
    filename : str
        name of file.

    Returns
    -------
    avg_rho : array
        average number of final recovered nodes.

    '''
    args = []
    for itr in np.arange(simulation):
         args.append([itr,G,alphas,mu,seeds])
         
    pool = Pool(processes = 8)
    results = pool.map(cg.run_SIR_mseed,args)
    rhos = np.array(results)
    avg_rho = np.average(rhos,axis=0)
    
    return avg_rho 

def SpreadNRInfluence(resultpath,G,nseed):
    
    simulation = 1000
    mu = 1
    alphas = np.arange(0,1.01,0.01)
    nrhos={}
    
    for n in nseed.keys():
        print('n',n)
        clique_rhos = {}
        for clique in nseed[n].keys():
            print('clique:',clique)
            infecteds = nseed[n][clique]
            filename = str(n)+'_seed_'+str(clique)+'_clique'
            rhos= Spreadmseed(G,alphas,mu,infecteds,simulation,resultpath,filename)
            clique_rhos[clique] = rhos
        nrhos[n] = clique_rhos 
        
    SaveIM(nrhos,resultpath,'NR5_nrhos')
    
    return nrhos

def NseedSpread(networkpath,resultpath,G,netIndex,clique):
    '''
    spread with multiply seeds for different centralities on original network 

    Parameters
    ----------
    networkpath : 
        path to load the network
    resultpath : str
        path to save the result
    G : graph
        network.
    netIndex : dict
        map realtion.
    clique : int
        value of k-clique.

    Returns
    -------
    Nspread : dict
        spreading with multiply seeds.

    '''
    #set the parameter 
    Nspread = {}
    alphas = np.arange(0,0.31,0.01)
    simulation = 1000
    mu = 1
    N = 51    
    
    #calculate the adaption centrality
    adaptNC = adaptNodeCent(networkpath,resultpath,G,netIndex,clique,N) #dict[n]
    adaptNC = cg.load(resultpath+'/IM/N_cseeds')

    for n in np.arange(1,N,1): 
        
        #spreading on original network with n seeds 
        print('selecting %d seeds to influence maximization'%n)
        Mseeds = adaptNC[n]
        method_spread = {}
        
        for mseed in Mseeds.keys():
            file =str(n)+'_ms_'+str(mseed)
            avg_rho = Spreadmseed(G,alphas,mu,Mseeds[mseed],simulation,resultpath,file)   
            method_spread[mseed] = avg_rho
        Nspread[n] = method_spread
    
    #save the results
    SaveIM(Nspread,resultpath,'NseedSpread')
    
    return Nspread

def RunIM(networkpath,resultpath,seed_n):
    '''
    
    calculate the influence maximization for different centrality with a given centrality
    
    Parameters
    ----------
    networkpath : str
        path to load the networks.
    resultpath : str
        path to save the results
    seed_n : int
        the number of initial seeds.

    Returns
    -------
    G : graph
        network.
    betas : array
        infection rate.
    Crhos_matrix_set : dict
        spread data for different centrality is saved as the matrix.
    cindex : dict
        map realtion between centrality and index.
    normMatrix : array
        normalized performance for different centrality, x \in [0,1].
    methodindex : array
        the best method to achieve the influence maximization.
    '''
    
    #load the network 
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)    
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])
    #PlotNwDist(path,netIndex)
    
    #identify nseeds by using the network reduction method
    #[nseed,npath]= NR_seed(networkpath,resultpath,G,netIndex)
    #spread on the original network with multiple seed
    #nrhos = SpreadNRInfluence(resultpath,G,nseed)

    #spread on the original network with multiple seed for different centralities
    #there are spread data in the files and please uncomment if run this code again 
    #nspread =  NseedSpread(networkpath,resultpath,G,netIndex,4)
    nspread = cg.load(resultpath+'/IM/NseedSpread')
    
    #transform the spread datasets
    betac = cg.CriticalInfectRate(G)
    betas = [betac,2*betac,3*betac,5*betac]
    [Crhos_matrix_set,cindex] = transform_Crhos(nspread,betas)
    [normMatrix,methodindex] = transform_CrhosBeta(G,nspread,seed_n,cindex,resultpath)
    
    return G,betas,Crhos_matrix_set,cindex,normMatrix,methodindex


def RunMultiDatasets(networkpath,resultpath,files):
    '''
    perform the numerical simulation of influencial maximization for each datasets

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the resultpath
    files : str
        filenames.

    Returns
    -------
    None.

    '''
    #set the number of seeds
    seed_n = 30
    
    for each in files:
        #for each datasets
        print('datasets:',each)
        eachNetpath = networkpath + '/' + each
        eachRespath = resultpath  + '/' + each
        
        #calculate the influence maximization for different centrality with a given centrality
        [G,betas,Crhos_matrix_set,cindex,normMatrix,methodindex]= RunIM(eachNetpath,eachRespath,seed_n)
    
        Crhos_array = normMatrix[12,:]

        SaveIM(cindex,eachRespath,'cindex')
        SaveIM(normMatrix,eachRespath,'normMatrixn'+str(seed_n))
        SaveIM(methodindex,eachRespath,'methodindex'+str(seed_n))
        SaveIM(Crhos_matrix_set,eachRespath,'Crhos_matrix_set')
        SaveIM(betas,eachRespath,'betas')
        SaveIM(G,eachRespath,'G')
        SaveIM(Crhos_array,eachRespath,'rhos_norm_n'+str(seed_n)+'_beta12')
        
def RankInDict(Dict):
    '''
    rank for the element in dict according to value of dict

    Parameters
    ----------
    Dict : dict
        a dict data needed to be ranked.

    Returns
    -------
    keysInx : dict 
        return the keys corresponding to inx.
    rankinx : array
        return the inx in a descending order.
    '''
    #record the edge index in the Reducted network and then generate the rank according to the edge Score   
    values= [] #values in dict
    keysInx = {} #keys index in dict
    for inx, (key,value) in enumerate(Dict.items()):
        keysInx[inx] = key 
        values.append(value)
    rankinx = np.argsort(-np.array(values)) #sort from large to small
    
    return keysInx, rankinx

def NREdge(path,G,netIndex,clique):
    
    #load the reducted network
    Redgelist = cg.load(path+'/'+str(netIndex[clique])+'_reducedEdgelist')
    Wnewnodes = cg.load(path+'/'+str(netIndex[clique])+'_Wnewnodes')
    #AllNodesMap = cg.ReducedNodeMap(path)
    AllEdgesMap = cg.ReducedEdgeMap(path)
    RG = cg.weightNetwork(Redgelist)
    
    #locate the important edges on macro-layer and build the map between reducted edge and original edges
    Sedge = {}
    RedgeToOld = {}
    for edge in list(RG.edges):
        stedge = (sorted(edge)[0],sorted(edge)[1])
        Oedge = cg.TrackBackEdges([stedge],AllEdgesMap)
        weight = RG.get_edge_data(edge[1],edge[0])['weight']
        if  weight != len(Oedge):
            raise Exception("Nedge != weight ! Weight:%d, Number of Edge:%d"%(weight,len(Oedge)))
        Sedge[stedge] = Wnewnodes[edge[0]] * Wnewnodes[edge[1]]  
        RedgeToOld[stedge] = Oedge
    
    #rank the edge according to the weight of edges
    [edgeIndex, rankinx] = RankInDict(Sedge)
    
    #generate the rank for each edge in the original network or on the micro-layer
    NR5 = []
    for rkinx in rankinx:
        groupSk = {}
        for oedge in RedgeToOld[edgeIndex[rkinx]]:
            sk = G.degree[oedge[0]] * G.degree[oedge[1]]
            groupSk[oedge] = sk
        [SkedgeIndex,Skrankinx] = RankInDict(groupSk)
        for inx in Skrankinx:
            NR5.append(SkedgeIndex[inx])
            
    return NR5    

def NREseed(networkpath, resultpath):
    
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)    
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])
    
    NR_Eseed ={}
    for clique in sorted(netIndex.keys())[:8]:
        print('clique',clique)
        NR = NREdge(networkpath,G,netIndex,clique)
        #SaveER(NR,path,str(clique+1)+'_NR')
        NR_Eseed[clique+1] = NR
    SaveER(NR_Eseed,resultpath,'NR_Eseed')
    
    ps = [0.01,0.05,0.10]
    pedges = {}
    for p in ps:
        Np = int(round(G.size()*p,0))
        Redge={}
        for clique in sorted(netIndex.keys())[:8]:
            Redge[clique+1] = NR_Eseed[clique+1][:Np]
        pedges[p] = Redge  
    
    SaveER(pedges,resultpath,'P_Eseed')
    

def Rlcc(subG):
    '''
    calculate the largest connected component of network

    Parameters
    ----------
    subG : graph
        network.

    Returns
    -------
    TYPE float
        the number of nodes.

    '''
    largest_cc = max(nx.connected_components(subG), key=len)
    return len(largest_cc)

def Dismantal(resultpath,file_name):
    '''
    calculate the lcc when the identified edges is iteratively removed

    Parameters
    ----------
    resultpath : str
        path to load the result.
    file_name : str
        file name.

    Returns
    -------
    None.

    '''
    edgelist = cg.load(resultpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])  
    NR = cg.load(resultpath+'/ER/'+file_name)         
    
    update_G = copy.deepcopy(G) 
    lcc ={}
    i = 0
    for i,edge in enumerate(NR):
        print('Nnum of removed edges:',i)
        rp = i/G.size()
        lcc[rp] = Rlcc(update_G)
        update_G.remove_edge(edge[0],edge[1])
        
    file = file_name.strip('_seed')
    SaveER(lcc,resultpath,file+'_lccs')
    
def EdgeRemove(networkpath,resultpath,clique,topNedge):
    '''
    Remove the edges identified by network reduction method

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    clique : int
        value of k-clique.
    topNedge : float
        the number of top-n edges.

    Returns
    -------
    None.

    '''
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)  
    AllEdgesMap = cg.ReducedEdgeMap(networkpath)
    
    Alledgescore ={}
    print('clique',clique)
    
    #load the reducted network
    redgelistInx= netIndex[clique]
    redgelist = cg.load(networkpath+'/'+str(redgelistInx)+'_reducedEdgelist') 
    RG = cg.weightNetwork(redgelist)
    Edgeweight = cg.load(networkpath+'/'+str(redgelistInx)+'_Wnewedges')
    Nodeweight = cg.load(networkpath+'/'+str(redgelistInx)+'_Wnewnodes')
        
    #score for each edge in reducted edge and original edges
    for nr_edge in Edgeweight.keys():
        if nr_edge[0] != nr_edge[1]:
            s = (Nodeweight[nr_edge[0]] * Nodeweight[nr_edge[1]])/Edgeweight[nr_edge]
            if Alledgescore.get(nr_edge) == None: 
                Alledgescore[nr_edge] = s
                
    #rank for all edges on the basis of score
    [keys,ranki]= RankInDict(Alledgescore)
    rank_edge = [keys[ranki[i]] for i in np.arange(len(ranki))]
    
    NR5 = []
    #generating the order of removed edges according to maximum node weight 
    for i,redge in enumerate(rank_edge):
        #print('i',i)        
        cuts = nx.algorithms.connectivity.cuts.minimum_edge_cut(RG,redge[0],redge[1])
        for edge_cut in list(cuts):
           redge = (sorted(edge_cut)[0],sorted(edge_cut)[1])
           Oedge = cg.TrackBackEdges([redge],AllEdgesMap)
           for oedge in Oedge:
              if oedge not in NR5:
                  NR5.append(oedge)
        if len(NR5) > topNedge:
            break
        
    SaveER(NR5,resultpath,'NR'+str(clique)+'_topN_seed')
    Dismantal(resultpath, 'NR'+str(clique)+'_topN_seed')

def BetweenAdp(networkpath,resultpath,topNedge):
    '''
    iteratively remove the edges with highest betweenness centrality

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    topNedge : int
        the number of removed edges.

    Returns
    -------
    None.

    '''
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])
    update_G = copy.deepcopy(G) 
    betwseed = []
    lccs ={}
    #PathLs = {}
    
    for num in np.arange(0,topNedge):
        print('num',num)
        
        #calculate the edge betweenness 
        edgeBetween = nx.edge_betweenness_centrality(update_G)
        [keysInx, rankinx] = RankInDict(edgeBetween)
        seed = keysInx[rankinx[0]] #locate the edge seed
        betwseed.append(seed)
        
        #remove the current maximum edge 
        lccs[num] = Rlcc(update_G)
        update_G.remove_edge(seed[0],seed[1])
    
    SaveER(betwseed,resultpath,'Betw_topN_seed')
    SaveER(lccs, resultpath,'Betw_topN_lccs')

def Between(networkpath,resultpath,topNedge):
    '''
    remove the edges identified by the centrality of betweenness 

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    topNedge : int
        the number of edges.

    Returns
    -------
    None.

    '''
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])
    edgeBetween = nx.edge_betweenness_centrality(G)
    [keysInx, rankinx] = RankInDict(edgeBetween)
    betwseed = [keysInx[rankinx[i]] for i in np.arange(0,topNedge)] #locate the edge seed
     
    SaveER(betwseed,resultpath,'BetwO_topN_seed')
    Dismantal(resultpath,'BetwO_topN_seed')
    
def edgedegree(G):
    '''
    calculate score of edges by k1*k2

    Parameters
    ----------
    G : graph
        network.

    Returns
    -------
    Edgek : dict
        edge score calculated by k1*k2.
    '''
    
    Edgek = {}
    for edge in G.edges():
        Edgek[edge] = G.degree[edge[0]] * G.degree[edge[1]]
    
    return Edgek

def DegreeAdp(networkpath,resultpath,topNedge):
    '''
    iteratively remove the edges with highest degree centrality

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    topNedge : int
        the number of edges.

    Returns
    -------
    None.

    '''
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])
    update_G = copy.deepcopy(G) 
    kseed = []
    lccs ={}
    
    for num in np.arange(0,topNedge):
        
        #calculate the edge score by k1*k2 
        edgek = edgedegree(update_G)
        [keysInx, rankinx] = RankInDict(edgek)
        seed = keysInx[rankinx[0]] #locate the edge seed
        kseed.append(seed)
        
        rm = num/G.size()
        lccs[rm] = Rlcc(update_G)
        update_G.remove_edge(seed[0],seed[1])
    
    SaveER(kseed,resultpath,'k_topN_seed')
    SaveER(lccs, resultpath,'k_topN_lccs')
    
def Degree(networkpath,resultpath,topNedge):
    '''
    remove the edges with highest degree centrality

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    topNedge : int
        the number of edges.

    Returns
    -------
    None.

    '''
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])  
    kseed = []
    
    #calculate the edge betweenness 
    edgek = edgedegree(G)
    [keysInx, rankinx] = RankInDict(edgek)
    kseed = [keysInx[rankinx[i]] for i in np.arange(0,topNedge)] #locate the edge seed
     
    SaveER(kseed,resultpath,'kO_topN_seed')
    Dismantal(resultpath,'kO_topN_seed')    
    
def Spreadseed(G,alphas,mu,simulation,path,filename):
    '''
    spread on network with multiply seeds

    Parameters
    ----------
    G : graph
        network.
    alphas : array
        infection probability.
    mu : float
        recovery probability.
    simulation : int
        simulation times.
    path : str
        path to save the result.
    filename : str
        filename.

    Returns
    -------
    avg_rho : array
        the number of recovered nodes.

    '''
    args = []
    for itr in np.arange(simulation):
         seeds = random.choice(list(G.nodes()))
         args.append([itr,G,alphas,mu,seeds])
         
    pool = Pool(processes = 8)
    results = pool.map(cg.run_SIR_mseed,args)
    rhos = np.array(results)
    avg_rho = np.average(rhos,axis=0)
    
    pk.dump(rhos,open(path+'/'+str(filename),'wb'))
    np.savetxt(path+'/'+str(filename)+'.csv',avg_rho,delimiter=',')
    
    return avg_rho     

def ERspread(networkpath,resultpath,filename,betas,Nredge):
    '''
    spread on network with removed edges

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to load the result.
    filename : str
        file name.
    betas : array
        infection rate.
    Nredge : int
        the number of removed edges.

    Returns
    -------
    None.

    '''
    mu = 1
    simulation = 1000
    
    #load the initial seeds
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])  
    rseed = cg.load(resultpath+'/ER/'+filename)
    
    #creat the file name and directory
    file = filename.strip('_seed')   
    spread_path = resultpath+'/ER/'+file +'_spread'
    if not os.path.exists(spread_path):
       os.mkdir(spread_path)
      
    #begin to spread
    update_G = copy.deepcopy(G)
    for num,edge in enumerate(rseed[:Nredge]):
        print('num/Nredge:',num)
        Spreadseed(update_G,betas,mu,simulation,spread_path,num)
        update_G.remove_edge(edge[0],edge[1])  

def ERspreads(path):
    
    betas = np.arange(0,1.01,0.01)
    Nredge = 100
    
    #Began to spread for NR
    ERspread(networkpath,resultpath,'NR5_topN_seed',betas,Nredge)
    
    #Began to spread for k
    ERspread(networkpath,resultpath,'k_topN_seed',betas,Nredge)
    
    #Began to spread for kO
    ERspread(networkpath,resultpath,'kO_topN_seed',betas,Nredge)
    
    #Began to spread for Betw
    ERspread(networkpath,resultpath,'Betw_topN_seed',betas,Nredge)
    
    #Began to spread for Betwo
    ERspread(networkpath,resultpath,'Betwo_topN_seed',betas,Nredge)
    
def RunEdgeImmune(networkpath,resultpath,files):
    '''
    run the edge immune for four datasets

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save and load the result.
    files: list
        datasets
        
    Returns
    -------
    None.

    '''
    
    for each in files:
        
        #for each datasets
        print('datasets:',each)
        eachNetpath = networkpath + '/' + each
        eachRespath = resultpath  + '/' + each
        
        #identify the important edges by network reduction method
        #NREseed(eachNetpath,eachRespath)
        
        #set the parameter
        clique = 4
        topNedge = 100
        #remove the edges identified by network reduction method
        EdgeRemove(eachNetpath,eachRespath,clique,topNedge)
        
        #remove the edges by adaptive edge betweenness and edge betweenness  
        BetweenAdp(eachNetpath,eachRespath,topNedge)
        Between(eachNetpath,eachRespath,topNedge)
        
        #remove the edges by adaptive degree and degree
        DegreeAdp(eachNetpath,eachRespath,topNedge)
        Degree(eachNetpath,eachRespath,topNedge)
    
        ERspreads(eachNetpath,eachRespath)
        
    #PlotLcc(path)
    #Nredge = 100
    #PlotER(path,Nredge)
    
def NR10(networkpath,resultpath,G,netIndex,clique,N):
    '''
    identify the Sentinel Surveillance by the 10-clique CGNs 

    Parameters
    ----------
    networkpath : str
        path to load the network .
    resultpath : str
        path to save the result.
    G : graph
        network.
    netIndex : dict 
        map relation between index and network.
    clique : int
        value of clique.
    N : int
        the number of seed nodes .

    Returns
    -------
    None.

    '''
    edgelist = cg.load(networkpath+'/'+str(netIndex[clique])+'_reducedEdgelist')
    G_reducted = cg.weightNetwork(np.array(edgelist))
    
    #calculate the weight degree of supernodes 
    Nodeweight = cg.NodeWeightDegree(G_reducted)
    
    NRseeds={}
    for n in np.arange(1,N):
        NRseeds[n] = Identify_seed(networkpath,G,Nodeweight,n)
    
    SaveSS(NRseeds,resultpath,'NR10_seed')
    
def Aspreadmseed(G,alphas,mu,sentinels,simulation,path,filename):
    '''
    numerical simulation with multiple seed for sentinel surveillance   
    with parallel computation
    
    Parameters
    ----------
    G : graph
        network.
    alphas : array
        infection rate.
    mu : float
        recovery probability.
    sentinels : list 
        initial seed node.
    simulation : int
        simulation times.
    path : str
        path to save the results.
    filename : str
        file name.

    Returns
    -------
    avg_rho : array
        average number of recovered nodes.

    '''
    args = []
    for itr in np.arange(simulation):
         args.append([itr,G,alphas,mu,sentinels])
         
    pool = Pool(processes = 20)
    results = pool.map(cg.run_SIR_Sentinel_mseed,args)
    rhos = np.array(results)
    avg_rho = np.average(rhos,axis=0)
    
    SaveSSspread(rhos,path,filename)
    np.savetxt(path+'/SS/spread/'+filename+'.csv',avg_rho,delimiter=',')
    
    return avg_rho  

def Sentinel_Spread(resultpath,G):
    '''
    numerical simulation of sentinel surveillance with multiple seed for different centralities

    Parameters
    ----------
    resultpath : str
        path to load and save the result.
    G : graph
        network.

    Returns
    -------
    None.

    '''
    Ns_arhos = {}
    alphas = np.arange(0,0.31,0.01)
    simulation = 1000
    mu = 1
    N = 51    
  
    #daption centrality
    adaptNC = cg.load(resultpath+'/SS/N_cseeds')
    NRseed  = cg.load(resultpath+'/SS/NR10_seed')
    
    #replacing the seed nodes of NR5 by NR10 
    for n in np.arange(1,N,1):    
        NC_sentinel = adaptNC[n]
        for sentinel_strategy in NC_sentinel.keys():
            if sentinel_strategy == 'NR5':
               adaptNC[n][sentinel_strategy]= NRseed[n]
    
    #save the seed nodes for different centralities
    SaveSS(adaptNC,resultpath,'N_cseed')
           
    #start to spread with n initial seed 
    for n in np.arange(1,N,1): 
        
        print('selecting %d node as sentinel to spread'%n)
        NC_sentinel = adaptNC[n]
        S_aspread = {}
        
        for sentinel_strategy in NC_sentinel.keys():
            #for each centrality
            filename = str(n)+'_ss_'+str(sentinel_strategy)
            sentinels= NC_sentinel[sentinel_strategy]
            sentinel_avg_arho = Aspreadmseed(G,alphas,mu,sentinels,simulation,resultpath,filename)   
            S_aspread[sentinel_strategy] = sentinel_avg_arho
        
        Ns_arhos[n] = S_aspread
    
    #save the spread result for diffent number of initial seeds
    SaveSS(Ns_arhos,resultpath,'Ns_FirstTime')

def NRNodeSentinel(networkpath,G,netIndex,clique,N):
    
    edgelist = cg.load(networkpath+'/'+str(netIndex[clique])+'_reducedEdgelist')
    G_reducted = cg.weightNetwork(np.array(edgelist))
    Nodeweight = cg.NodeWeightDegree(G_reducted)

    NR5seeds={}
    for n in np.arange(1,N):
        NR5seeds[n] = Identify_seed(Identify_seed,G,Nodeweight,n)
        
    simulation = 1000
    alphas = np.arange(0,1.01,0.01)
    mu = 1

    Ns_arhos={}
    for n in NR5seeds.keys():
        file = str(n)+'_ss_NR5'
        sentinels = NR5seeds[n]
        sentinel_avg_arho = Aspreadmseed(G,alphas,mu,sentinels,simulation,resultpath,file)   
        Ns_arhos[n] = sentinel_avg_arho
    
    SaveSS(Ns_arhos,resultpath,'NR10_firstTime')
    
def RunSentinelSurveillance(networkpath,resultpath,files):
    '''
    run the sentinel surveillance for four datasets

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save and load the result.
    files: list
        datasets
        
    Returns
    -------
    None.

    '''
    
    for each in files:
      
        #for each datasets
        print('datasets:',each)
        
        eachNetpath = networkpath + '/' + each
        eachRespath = resultpath  + '/' + each
    
        #load the basic infomation   
        sizeCliques = np.loadtxt(eachNetpath+'/sizeCliques.txt')
        [netIndex,Nweight] = cg.NetworkInx(sizeCliques,eachNetpath)    
        edgelist = cg.load(eachNetpath+'/0_reducedEdgelist') 
        G = cg.load_network(edgelist[:,0:2])
        
        #set the parameter
        N = 51
        clique = 9
        
        #identify the Sentinel Surveillance by network reduction method
        NR10(eachNetpath,eachRespath,G,netIndex,clique,N)
        Sentinel_Spread(eachRespath,G)
        #NRNodeSentinel(networkpath,G,netIndex,clique,N)    

def truncate_colormap(colormap, minval=0.0, maxval=1.0, n=100):
    '''
    truncate the colormap
    
    Parameters
    ----------
    colormap : LinearSegmentedColormap
        colormap.
    minval : TYPE, optional
        minimum value. The default is 0.0.
    maxval : TYPE, optional
        maximum value. The default is 1.0.
    n : int, optional
        interval. The default is 100.

    Returns
    -------
    new_cmap : LinearSegmentedColormap
         new colormap.

    '''
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("trunc({n},{a:.2f},{b:.2f})".format(n=colormap.name, a=minval, b=maxval)
        ,colormap(np.linspace(minval, maxval, n)))
    
    return new_cmap

def PlotSchematic1(ax, figurepath):
    
     #plot the schematic 
     figure1 = mpimg.imread(figurepath + '/figure1.png')
     ax.imshow(figure1)
     ax.axis('off')
     cg.PlotAxes(ax,'','','a')
     
def PlotSchematic2(ax, figurepath):
     
     #plot the schematic 
     figure2 = mpimg.imread(figurepath+'/figure2.png')
     ax.imshow(figure2)
     ax.axis('off')
     cg.PlotAxes(ax,'','','e')
     
def PlotSchematic3(ax, figurepath):
     
     #plot the schematic     
     figure3 = mpimg.imread(figurepath+'/figure3.png')
     ax.imshow(figure3)
     ax.axis('off')
     cg.PlotAxes(ax,'','','i')
     
def PlotMethodNseed(ax,G,betas,Crhos_matrix_set,cindex,colors):
    '''
    IM for different method plotted as a function of number of seed
    
    '''
    fontsize = 20
    n_legend = 18
    font_label = {'family': "Arial", 'size':fontsize}
    
    beta = betas[1]
    rhos = Crhos_matrix_set[beta]/G.order()
    markers = ['o','^','s','<','h','+','H','>','1','<']
    
    rhos_df = pd.DataFrame()
    rhos_df['NumofNode'] = np.arange(1,51)
    
    for i,centra in enumerate(cindex.keys()):
        nrho = rhos[:,cindex[centra]][1:]
        rhos_df[centra] = nrho
        
    for i,centra in enumerate(cindex.keys()):
        if centra == 'NR5':
            sns.regplot(x='NumofNode',y=centra,data=rhos_df,x_bins=9, x_ci="sd",color='black',marker=markers[i],order=3,ax=ax,label=r'$NR_n(5)$')
        else:
            sns.regplot(x='NumofNode',y=centra,data=rhos_df,x_bins=9, x_ci="sd",color=colors(i-1),marker=markers[i],order=3,ax=ax,label='A. ' + centra)
    
    ax.text(2,0.158,r'GrQc',color='gray',fontdict=font_label)
    ax.text(2,0.15,r'$\beta=0.12$',color='gray',fontdict=font_label)
    ax.set_xlim(-1,51)
    ax.legend(loc='lower center', bbox_to_anchor=(0.35,-0.02,0.4,0.4),ncol=2,framealpha=0, fontsize=n_legend,borderpad=0.0,handlelength=1.0)
    cg.PlotAxes(ax,'$n$',r'$\rho$', 'b')

def PlotCompareMethod(ax,normMatrix,cindex,methodindex,color_map):
    '''
    compare NR with different method 
    
    '''
    fontsize= 20
    xticklabels = [r'$NR_n$','K','K.c.','B.t.','C.s.','E.g.','K.z','S.g.','CI','NB']
    yticks = np.arange(1,20,2)-0.5 
    x = np.arange(1,20,1)
    maxCentrality = [xticklabels[i] for i in methodindex[x]]
    print(maxCentrality)
    yticklabels = list(map(lambda x: round(x,2),np.arange(0.01,0.2,0.02)))
    
    color = plt.get_cmap(color_map)
    font_label = {'family': "Arial", 'size':fontsize}
    h=sns.heatmap(normMatrix[x],cmap=color,ax=ax,linewidths=.1,cbar=False)
    cb = plt.colorbar(h.collections[0],ax=ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\rho/\rho^{max}$',size=fontsize)

    ax.set_xticklabels(xticklabels,rotation=55,fontdict=font_label)
    ax.set_xlim(0,11.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,rotation=0,fontdict=font_label)
    
    rectangle1 = pc.Rectangle((10.1,0), 1, 1,color='#c9e4ca')
    rectangle2 = pc.Rectangle((10.1,1), 1, 2,color='#d8f3dc')
    rectangle3 = pc.Rectangle((10.1,2), 1, 3,color='#c9e4ca')
    rectangle4 = pc.Rectangle((10.1,3), 1, 4,color='#d8f3dc')
    rectangle5 = pc.Rectangle((10.1,4), 1, 19,color='#95d5b2')
    ax.add_artist(rectangle1)
    ax.add_artist(rectangle2)
    ax.add_artist(rectangle3)
    ax.add_artist(rectangle4)
    ax.add_artist(rectangle5)

    ax.text(10.3,0.55,'K',rotation='vertical',size=12,color='black')
    ax.text(10.3,1.8,'S.g.',rotation='vertical',size=12,color='black')
    ax.text(10.3,2.5,'K',rotation='vertical',size=12,color='black')
    ax.text(10.3,3.78,'K.z',rotation='vertical',size=12,color='black')
    ax.text(10.3,11.5,r'$NR_n$',rotation='vertical',size=12,color='black')
                
    cg.PlotAxes(ax,'',r'$\beta$',title='c')
    
def Spread_matrix(NseedSpread):
    
    #transform the NseedSpread into matrix format
    
    N = len(NseedSpread.keys())
    cindex = {c:i for i,c in enumerate(NseedSpread[1].keys())}
    matrix = np.zeros((31,len(cindex),N))
    for z,n in enumerate(NseedSpread.keys()):
        s = NseedSpread[n]
        for c in s.keys():
            matrix[:,cindex[c],z] = s[c]
    
    return matrix

def PlotsubAxes(ax):
    '''
    decorate the axes 
    '''    
    fontsize  = 20
    font_label = {'family': "Calibri", 'size':fontsize}
    
    ax.set_xlabel('$n$',fontdict = font_label)
    ax.set_ylabel(r'$\beta$',fontdict = font_label)
   
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)

def demo_grid_IM(fig,loc,spread_dict,color):
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """
    fontsize  = 20
    font_label = {'family': "Arial", 'size':fontsize}
    dataset_name = list(spread_dict.keys())
    
    #color = plt.get_cmap('GnBu')#viridis,
    grid = ImageGrid(fig, loc,  # similar to subplot(143)
                     nrows_ncols=(2, 2),
                     axes_pad=0.2,
                     label_mode="L",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad="1%",
                     cbar_set_cax=True
                     )
    extent = (0, 50, 0, 50)
    for i, (ax, cax) in enumerate(zip(grid, grid.cbar_axes)):
        names = dataset_name[i]
        im = ax.imshow(spread_dict[names], cmap=color,extent=extent)
        cb = cax.colorbar(im)
        cax.toggle_label(True)
        PlotsubAxes(ax)
        ax.text(10,10,names, fontdict = font_label)
        if i==0:
            cb.set_label(label=r'$\rho^{NR_{n}(5)}/\rho^{max}$',loc='center',fontdict = font_label)
            cb.ax.tick_params(labelsize=18)
            cb_min = round(np.min(spread_dict[names]),1)
            cb.set_ticks([cb_min,1])

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([5, 25, 45])
    grid.axes_llc.set_yticks([6/31*50, 16/31*50, 26/31*50])
    grid.axes_llc.set_xticklabels([5,25,45])
    grid.axes_llc.set_yticklabels([0.25,0.15,0.05])

def demo_grid_ER(fig,loc,spread_matrix,titles,color):
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """
    fontsize  = 20
    font_label = {'family': "Arial", 'size':fontsize}
    
    #color = plt.get_cmap('GnBu')#viridis,
    grid = ImageGrid(fig, loc,  # similar to subplot(143)
                     nrows_ncols=(2, 2),
                     axes_pad=0.2,
                     label_mode="L",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad="1%",
                     cbar_set_cax=True
                     )
    extent = (0, 100, 1, 100)
    for i, (ax, cax) in enumerate(zip(grid, grid.cbar_axes)):
        im = ax.imshow(spread_matrix[i], cmap=color,extent=extent)
        cb = cax.colorbar(im)
        cax.toggle_label(True)
        PlotsubAxes(ax)
        cb_max = round(np.max(spread_matrix[i]),1)
        ax.text(20,20,titles[i], fontdict = font_label)
        if i==0:
            cb.set_label(label=r'$\rho^{NR_{e}(5)}/\rho^{min}$',loc='center',fontdict = font_label)
            cb.ax.tick_params(labelsize=18)
            cb.set_ticks([1,cb_max])

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([10, 50, 90])
    grid.axes_llc.set_yticks([10, 50, 90])
    grid.axes_llc.set_xticklabels([10,50,90])
    grid.axes_llc.set_yticklabels([0.90,0.50,0.1])

def demo_grid_SS(fig,loc,spread_matrix,titles,color):
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """
    fontsize  = 20
    font_label = {'family': "Arial", 'size':fontsize}
    
    #color = plt.get_cmap('GnBu')#viridis,
    grid = ImageGrid(fig, loc,  # similar to subplot(143)
                     nrows_ncols=(2, 2),
                     axes_pad=0.2,
                     label_mode="L",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad="1%",
                     cbar_set_cax=True
                     )
    extent = (0, 50, 0, 50)
    for i, (ax, cax) in enumerate(zip(grid, grid.cbar_axes)):
        im = ax.imshow(spread_matrix[i], cmap=color,extent=extent)
        cb = cax.colorbar(im)
        cax.toggle_label(True)
        PlotsubAxes(ax)
        cb_max = round(np.max(spread_matrix[i]),1)
        ax.text(10,10,titles[i], fontdict = font_label)
        if i==0:
            cb.set_label(label=r'$T_{fd}^{NR_{s}(10)}/T_{fd}^{min}$',loc='center',fontdict = font_label)
            cb.ax.tick_params(labelsize=18)
            cb.set_ticks([1,1.16])

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([5, 25, 45])
    grid.axes_llc.set_yticks([6/31*50, 16/31*50, 26/31*50])
    grid.axes_llc.set_xticklabels([5,25,45])
    grid.axes_llc.set_yticklabels([0.25,0.15,0.05])
    
def PlotIMDatasets(fig,ax,datas_betaindex,color):
    '''
    plot the normlized performance of network reduction method on four datasets    
    
    '''
    ax.axis('off')
    norms_rhos ={}
    for dataset_name in datas_betaindex.keys():
        rhos = datas_betaindex[dataset_name]
        rhos_MAX = np.max(rhos,axis=1)
        norm_nr5 = rhos[:,0,:]/rhos_MAX
        norms_rhos[dataset_name] = norm_nr5
    
    demo_grid_IM(fig,(3,4,4),norms_rhos,color)
    cg.PlotAxes(ax, '', '','d')
    
def ParseRmER(path,filename,betacs,Nredge,G):
    '''
    load the rhos for different centralities
    
    '''
    rho_betacs = {}
    for beta_inx in betacs:
        rho_betac = {} 
        for fn in np.arange(Nredge):
            rhos = np.loadtxt(path+'/ER/' + filename + '/'+str(fn) +'.csv')
            rho_betac[fn] = rhos[int(beta_inx)]/G.order()
        rho_betacs[str(beta_inx)] = rho_betac
    
    pk.dump(rho_betacs, open(path+'/ER/' + filename + '/rhos_betac','wb'))
    
def PlotER(resultpath,ax,colors,font_label):
     '''
    Plot the result of edge immune for different centralities vs the number of seeds 

    Parameters
    ----------
    resultpath : str
        path to load the result.
    ax : axes
        axes.
    colors : list
        colors.
    font_label : dict
        set the font.

    Returns
    -------
    None.

    '''
     n_legend = 18     
     path = resultpath + '/GrQc' 
     #edgelist = cg.load(path+'/0_reducedEdgelist') 
     #G = cg.load_network(edgelist[:,0:2])  
     #betac = cg.CriticalInfectRate(G)
     #betacs = [int(round(beta,2)*100) for beta in [6*betac]]
     #Nredge = 100

     #ParseRmER(path,'NR5_topN',betacs,Nredge,G)
     #ParseRmER(path,'k_topN',betacs,Nredge,G)
     #ParseRmER(path,'kO_topN',betacs,Nredge,G)
     #ParseRmER(path,'Betw_topN',betacs,Nredge,G)
     #ParseRmER(path,'Betwo_topN',betacs,Nredge,G)
     filename = ['k_topN','kO_topN','Betw_topN','Betwo_topN','NR5_topN']
    
     rhos = pd.DataFrame()
     rhos['NumofEdge'] = np.arange(0,100)

     for i,file in enumerate(filename):
        rhos_betac = cg.load(path+'/ER/'+file+'/rhos_betac')
        rho_c = pd.DataFrame.from_dict(rhos_betac,orient='index').T
        rho_c=rho_c.rename(columns={'35':file})
        rhos = pd.concat([rhos,rho_c],axis=1)
     
     sns.regplot(x='NumofEdge',y='kO_topN',data=rhos,x_bins=9,x_ci='sd',ax=ax,marker='^',color=colors(0),label=r'K ($k_1 \times k_2$)',order=2)#,,label = file
     sns.regplot(x='NumofEdge',y='k_topN',data=rhos,x_bins=9,x_ci='sd',ax=ax,marker='s',color=colors(1),label='A. K',order=2)#,,label = file
     sns.regplot(x='NumofEdge',y='Betwo_topN',data=rhos,x_bins=9,x_ci='sd',ax=ax,marker='<',color=colors(2),label='Betwn.',order=2)#,,label = file
     sns.regplot(x='NumofEdge',y='Betw_topN',data=rhos,x_bins=9,x_ci='sd',ax=ax,marker='h',color=colors(3),label='A. Betwn.',order=2)#,,label = file
     sns.regplot(x='NumofEdge',y='NR5_topN',data=rhos,x_bins=9,x_ci='sd',ax=ax,marker='o',color='black',label=r'$NR_{e}(5)$',order=2)#,,label = file
     cg.PlotAxes(ax,'$n$',r'$\rho$',title='f')
     
     ax.text(76,0.340,'GrQc',color='gray',fontdict=font_label)
     ax.text(76,0.335,r'$\beta$ = 0.35',color='gray',fontdict=font_label)
     ax.set_xlim(0,100)
     ax.legend(loc='best', framealpha=0, fontsize=n_legend)

def ParseBetaER(path,filename,Nedge,G):
    '''
    Parse the result of edge immune for a given number of removed edges 

    Parameters
    ----------
    path : str
        path to load the results.
    filename : list
        name of centralities.
    Nedge : int
        the number of removed edges.
    G : graph
        networks.

    Returns
    -------
    None.

    '''
    rhos = pd.DataFrame(columns=filename)
    for i,file in enumerate(filename):
        rhos_rm50 = np.loadtxt(path+'/ER/'+file+'/'+str(Nedge)+'.csv')
        rhos_rm50_1d = rhos_rm50/G.order() #np.reshape(rhos_rm50,(101*1000))
        rhos[file] = rhos_rm50_1d
        
    rhos_p = pd.DataFrame(columns=filename)
    beta = [round(each,2) for each in np.arange(0,1.01,0.01)]#*1000
    for i,b in enumerate(beta):
       rhos_p = rhos_p.append(rhos.iloc[i]/rhos.min(axis=1)[i])
    
    filenames = filename.copy()
    filenames.append('beta')
    rhos_p['beta']=beta
    SaveER(rhos_p,path,'rhos_rm50')
     
def PlotERbeta(resultpath,ax,G,colors,font_label):
    '''
    Plot the result of edge immune for different centralities vs the infection rate
    
    Parameters
    ----------
    resultpath : str
        path to load the result.
    ax : axes
        axes.
    G : graph
        network.
    colors : list
        color.
    font_label : dict
        set the font.

    Returns
    -------
    None.

    '''

    Nredge = 80
    filename = ['k_topN','kO_topN','Betw_topN','Betwo_topN','NR5_topN']
    path = resultpath + '/GrQc' 
    ParseBetaER(path,filename,Nredge,G)
    rhos_rm50 = cg.load(path+'/ER/rhos_rm50') 
   
    sns.regplot(data=rhos_rm50,x='beta',y='kO_topN',ax=ax,x_bins=9,x_ci='sd',marker='^', color = colors(0),order=2)
    sns.regplot(data=rhos_rm50,x='beta',y='k_topN',ax=ax,x_bins=9,x_ci='sd',marker='s',color = colors(1),order=2)
    sns.regplot(data=rhos_rm50,x='beta',y='Betwo_topN',ax=ax,x_bins=9,x_ci='sd',marker='<',color = colors(2),order=2)
    sns.regplot(data=rhos_rm50,x='beta',y='Betw_topN',ax=ax,x_bins=9,x_ci='sd',marker='h',color = colors(3),order=2)
    sns.regplot(data=rhos_rm50,x='beta',y='NR5_topN',ax=ax,x_bins=9,x_ci='sd',marker='o', color = 'black',order=2)

    cg.PlotAxes(ax, r'$\beta$',r'$\rho/\rho^{min}$','g')
    ax.text(0.79,1.57,'GrQc',color='gray',fontdict=font_label)
    ax.text(0.79,1.52,r'$n$ = 80',color='gray',fontdict=font_label)
    ax.set_xlim(0,1)
     
def LoadERspread(path):
    '''
    load the result of spread about edge removed 
    
    '''
    localfile = ['Betw_topN','BetwO_topN','k_topN','KO_topN','NR5_topN']
    
    ns = np.arange(0,100)
    rhos = np.zeros((101,100,5))
    for z,lf in enumerate(localfile):
        load_path = path+'/'+lf
        for n in ns:
            rhos[:,n,z] = np.loadtxt(load_path+'/'+str(n)+'.csv')
        
    return rhos     

def PlotERDatasets(fig,ax,resultpath,files,color):
     '''
     Plot the result on different datasets
     
     '''
     ax.axis('off')
     titles = ['GrQC','CondMat','HepPh','NetScience']
        
     data_rhos = {}
     for file in files:
         path = resultpath + '/' + file + '/ER'
         data_rhos[file] = LoadERspread(path)
     
     dict_spread ={}
     for k,dataname in enumerate(data_rhos.keys()):
         spread = data_rhos[dataname]
         
         #s = np.argmin(spread,axis=2)
         spread_MIN = np.min(spread,axis=2)
         norm_spread = np.zeros((101,100,5))
         for i in np.arange(5):
             norm_spread[:,:,i] = spread[:,:,i]/spread_MIN
        
         dict_spread[k] = norm_spread[:,:,4]

     demo_grid_ER(fig,(3,4,8),dict_spread,titles,color)    
     cg.PlotAxes(ax, '', '','h')

def Spread_matrix100(NseedSpread):
    '''
    transform the dataset into matrix format
    
    '''
    N = len(NseedSpread.keys())
    cindex = {c:i for i,c in enumerate(NseedSpread[1].keys())}
    matrix = np.zeros((101,len(cindex),N))
    for z,n in enumerate(NseedSpread.keys()):
        s = NseedSpread[n]
        for c in s.keys():
            matrix[:,cindex[c],z] = s[c]
    
    return cindex,matrix


def PlotSS(resultpath,ax,font_label):
    '''
    Plot the sentinel surveillance versus the number of initial seeds
    '''
    
    betainx = 12
    
    #transform the NR method dataset into matrix
    NR8_spread = cg.load(resultpath+'/GrQc/SS/NR10_firstTime')
    NR8_spread_matrix = np.zeros((101,50))
    for i,n in enumerate(NR8_spread.keys()):
        NR8_spread_matrix[:,i]= NR8_spread[n]
    
    #transform the all method dataset into dataframe
    Nspread = cg.load(resultpath+'/GrQc/SS/Ns_FirstTime')#Ns_aspread
    [cindex,Nspread_matrix] = Spread_matrix100(Nspread)
    rhos_ss = pd.DataFrame()
    rhos_ss['NumofNode'] = np.arange(1,51)
    for i,centra in enumerate(cindex.keys()):
        if centra == 'NR5':
            rhos_ss[centra] = NR8_spread_matrix[betainx,:]
        else:
            nrho = Nspread_matrix[betainx,cindex[centra],:]
            rhos_ss[centra] = nrho
    
    #Plot the figure 
    colors = plt.get_cmap('Paired')
    markers = ['o','^','s','<','h','+','H','>','1','<']
    
    for i,c in enumerate(cindex.keys()):
        if c == 'NR5':
            sns.regplot(x='NumofNode',y=c,data=rhos_ss,x_bins=9, x_ci="sd",color='black',marker=markers[i],order=2,ax=ax,label=r'$NR_s(10)$')
        else:
            sns.regplot(x='NumofNode',y=c,data=rhos_ss,x_bins=9, x_ci="sd",color=colors(i-1),marker=markers[i],order=2,ax=ax,label='A. ' + c)

    ax.legend(loc='upper center', bbox_to_anchor=(0.385,0.5,0.4,0.4),ncol=2,framealpha=0, fontsize=18,borderpad=0.0,handlelength=1.0)
    ax.text(2,1.9,'GrQc',color='gray',fontdict=font_label)
    ax.text(2,1.8,r'$\beta= 0.12$',color='gray',fontdict=font_label)
    ax.set_xlim(-1,51)    
    cg.PlotAxes(ax,'$n$',r'$T_{fd}$','j') 
    
def LoadSSspread(path,file):
    
    spread_dict = cg.load(path+'/'+'Ns_FirstTime')
    N = len(spread_dict.keys())
    cindex = {c:i for i,c in enumerate(spread_dict[1].keys())}
    matrix = np.zeros((31,N,len(cindex)))
    
    if file == 'GrQC':
       spread_NR10 = cg.load(path+'/NR10_firstTime')
       for z,n in enumerate(spread_dict.keys()):
           s = spread_dict[n]
           for c in s.keys():
               if c == 'NR5':
                  matrix[:,z,cindex[c]] = spread_NR10[n][:31]
               else:
                  matrix[:,z,cindex[c]] = s[c][:31] 
    else:
        for z,n in enumerate(spread_dict.keys()):
            s = spread_dict[n]
            for c in s.keys():
                matrix[:,z,cindex[c]] = s[c]
    return matrix

def PlotSSDatasets(fig,ax,root_path,files,color):
    '''
    
    Plot the result of SS on different datasets

    '''
    ax.axis('off')
    titles = ['GrQC','CondMat','HepPh','NetScience']
    data_rhos = {}
    
    for file in files:
        path = root_path + '/' + file + '/SS'
        data_rhos[file] = LoadSSspread(path,file)
        
    data_norms = {}
    for k,dataname in enumerate(data_rhos.keys()):
         spread = data_rhos[dataname]
         
         #s = np.argmin(spread,axis=2)
         spread_MIN = np.min(spread,axis=2)
         norm_spread = np.zeros((31,50,10))
         for i in np.arange(10):
             norm_spread[:,:,i] = spread[:,:,i]/spread_MIN
             
         data_norms[k] = norm_spread[:,:,0]
     
    demo_grid_SS(fig,(3,4,12),data_norms,titles,color)    
    cg.PlotAxes(ax,'','','l')
    
def PlotSSbeta(resultpath,ax,color_map): 
    '''
    Plot the sentinel surveillance versus different infection rate
    
    '''
    #load the data and network
    N =40
    NR10_spread = cg.load(resultpath+'/GrQc/SS/NR10_firstTime')
    Nspread = cg.load(resultpath+'/GrQc/SS/Ns_FirstTime')#Ns_aspread
    [cindex,Nspread_matrix] = Spread_matrix100(Nspread)
    
    #create the dataframe
    rhos_ss_beta = pd.DataFrame()
    rhos_ss = pd.DataFrame()
    #rhos_ss_beta['beta'] = np.arange(0,1.01,0.01)
    
    for i,centra in enumerate(cindex.keys()):
        if centra == 'NR5':
            rhos_ss_beta[centra] = NR10_spread[N]
        else:
            nrho = Nspread_matrix[:,cindex[centra],N]
            rhos_ss_beta[centra] = nrho
    
    best_centrality = rhos_ss_beta.idxmin(axis=1)
    standard_centrality = rhos_ss_beta.idxmin(axis=1)
    for i in np.arange(0,101,1):
       c = standard_centrality[i]
       rhos_ss.loc[i,rhos_ss_beta.columns] = rhos_ss_beta.loc[i]/rhos_ss_beta.loc[i,c]
    
    data = np.array(rhos_ss.loc[1:19])
    #fig,ax = plt.subplots(1,1,figsize=(6,7),tight_layout=True)
    color = plt.get_cmap(color_map) 
    
    fontsize= 20
    font_label = {'family': "Arial", 'size':fontsize}
    xticklabels = [r'$NR_s$','K','K.c.','B.t.','C.s.','E.g.','K.z','S.g.','CI','NB']
    yticks = np.arange(0,20,2)+0.5 
    yticklabels = list(map(lambda x: round(x,2),np.arange(0.01,0.20,0.02)))
    
    h=sns.heatmap(data,cmap=color,ax=ax,linewidths=0.1,cbar=False)

    cb = plt.colorbar(h.collections[0],ax=ax)
    cb.ax.tick_params(labelsize=18)
    cb.set_label(label=r'$T_{fd}/T_{fd}^{min}$',size=fontsize)

    ax.set_xticklabels(xticklabels,rotation=55,fontdict=font_label)
    ax.set_xlim(0,11.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,rotation=0,fontdict=font_label)
    
    #best_c = best_centrality[1:19]
    rectangle1 = pc.Rectangle((10.1,0), 1, 2,color='#d8f3dc')
    rectangle2 = pc.Rectangle((10.1,2), 1, 4,color='#c9e4ca')
    rectangle3 = pc.Rectangle((10.1,4), 1, 5,color='#d8f3dc')
    rectangle4 = pc.Rectangle((10.1,5), 1, 9,color='#95d5b2')
    rectangle5 = pc.Rectangle((10.1,9), 1, 10,color='#c9e4ca')
    rectangle6 = pc.Rectangle((10.1,10), 1, 18,color='#95d5b2')

    ax.add_artist(rectangle1)
    ax.add_artist(rectangle2)
    ax.add_artist(rectangle3)
    ax.add_artist(rectangle4)
    ax.add_artist(rectangle5)
    ax.add_artist(rectangle6)
    
    ax.text(10.3,1.2,'NB',rotation='vertical',size=12,color='black')
    ax.text(10.3,3.2,'CI',rotation='vertical',size=12,color='black')
    ax.text(10.3,4.7,'NB',rotation='vertical',size=12,color='black')
    ax.text(10.3,8,r'$NR_s$',rotation='vertical',size=12,color='black')
    ax.text(10.3,9.7,'K.z',rotation='vertical',size=12,color='black')
    ax.text(10.3,15,r'$NR_s$',rotation='vertical',size=12,color='black')

    cg.PlotAxes(ax,'',r'$\beta$','k') #Centralities 
    
def PlotProbApp(resultpath,figurepath, files):
    '''
    Plot the result of application to real propogation problem

    Parameters
    ----------
    resultpath : str
        path to load the result.
    figurepath : str
        path to save the figure.
    files : list
        datasets.

    Returns
    -------
    None.
    
    '''
    #set the color
    colors = plt.get_cmap('Paired') #set the discrete color 
    colormap = plt.get_cmap('GnBu')#'summer_r' set the continuous color
    trunc_cmap = truncate_colormap(colormap, 0.2, 0.8) #truncate the colormap
     
    #set the font
    fontsize = 20
    font_label = {'family': "Arial", 'size':fontsize}
    datas_betaindex = {}
    seed_n = 30
     
    #plot the figures
    fig,ax = plt.subplots(3,4,figsize=(25,21))
    plt.tight_layout(pad=6, h_pad= 8, w_pad= 8)
     
    #Plot the IM
    PlotSchematic1(ax[0,0],figurepath)

    for file in files:
         
       #load the result from each datasets
       path = resultpath+'/'+file+'/IM'
        
       #for GrQc dataset 
       if file == 'GrQC':
            
           #load the result
           cindex = cg.load(path+'/cindex')
           normMatrix = cg.load(path+'/normMatrixn'+str(seed_n))
           methodindex = cg.load(path+'/methodindex'+str(seed_n))
           Crhos_matrix_set = cg.load(path+'/Crhos_matrix_set')
           betas = cg.load(path+'/betas')
           G = cg.load(path+'/G')
            
           #compare NR with different adaptive method
           PlotMethodNseed(ax[0,1],G,betas,Crhos_matrix_set,cindex,colors)
           PlotCompareMethod(ax[0,2],normMatrix,cindex,methodindex,trunc_cmap)
           #PlotDiffMethod(ax[0,2],normMatrix,colors)
            
       NseedSpread = cg.load(path+'/NseedSpread')
       nspread_matrix = Spread_matrix(NseedSpread)
       #beta_index = Advantage_curve(nspread_matrix)
       datas_betaindex[file] = nspread_matrix
       
    #PlotMethodDatasets(ax[0,3],datas_betaindex,colors)
    PlotIMDatasets(fig,ax[0,3],datas_betaindex,trunc_cmap)
     
    #plot the ER
    PlotSchematic2(ax[1,0],figurepath)
    PlotER(resultpath,ax[1,1],colors,font_label)
    PlotERbeta(resultpath,ax[1,2],G,colors,font_label)
    PlotERDatasets(fig,ax[1,3],resultpath,files,trunc_cmap)
     
    #plot the SS
    PlotSchematic3(ax[2,0],figurepath)
    PlotSS(resultpath,ax[2,1],font_label)
    PlotSSbeta(resultpath,ax[2,2],trunc_cmap)
    PlotSSDatasets(fig,ax[2,3],resultpath,files,trunc_cmap)  

    plt.savefig(figurepath + '/figure5.png',dpi=300)
    plt.savefig(figurepath + '/figure5.pdf')
    plt.savefig(figurepath + '/figure5.eps')
    
if __name__ == '__main__':
    
  networkpath = root_path + '/NetworkReduction/fig3_reductionSize/network'  
  resultpath = root_path + '/NetworkReduction/fig5_application/result'
  figurepath = root_path + '/NetworkReduction/fig5_application/figure'
  theorypath = root_path + '/NetworkReduction/fig2_leastInfProb/result'
  
  #datasets 
  files = ['GrQC','CondMat','HepPh','NetScience']#'Email-Enron','Musae_facebook'Ã¯Â¼Å’'CondMat','HepPh','NetScience'
  
  #begin to run. 
  #please note that there are all results about spreaidng in existing file  
  #do not excute the 'Runxxx' function if only intend to plot the result, it will takes a lot of times to run the code and cover existing spreading result
  
  #influence Maximization
  #RunMultiDatasets(networkpath,resultpath,files)
  
  #Edge Immune
  #RunEdgeImmune(networkpath,resultpath,files)

  #Sentinel Surveillance
  #RunSentinelSurveillance(networkpath,resultpath,files)
    
  #plot the result: application to real propogation problems
  PlotProbApp(resultpath,figurepath, files)

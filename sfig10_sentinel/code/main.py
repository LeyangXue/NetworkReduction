# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:31:23 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle as pk 
from multiprocessing import Pool
import pandas as pd
import random 
import seaborn as sns

def Save(data,resultpath,filename):
    
    pk.dump(data,open(resultpath+'/'+filename,'wb'))
    
def Aspreadmseed(G,alphas,mu,sentinels,simulation,path,filename):
    
    args = []
    for itr in np.arange(simulation):
         args.append([itr,G,alphas,mu,sentinels])
         
    pool = Pool(processes = 20)
    results = pool.map(cg.run_SIR_Sentinel_mseed,args)
    rhos = np.array(results)
    avg_rho = np.average(rhos,axis=0)
    
    #SaveSS(rhos,path,filename)
    Save(rhos,path,filename)
    np.savetxt(path+'/'+filename+'.csv',avg_rho,delimiter=',')
    
    return avg_rho  
    
def findBlockSeed(G):
    
    maxk = 0
    degree = dict(nx.degree(G))
    
    for node in degree.keys():
        if degree[node] >  maxk:
           maxk = degree[node]
           maxkNode = node
           
    return maxkNode

def LocateSentinel(G,ReducedNodes,topN):#
    
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
        #seed.append(findBlockSentinel(blockG,between))
        seed.append(findBlockSeed(blockG))

    return seed
    
def identifyTopN(centrality,N):
    
    nodeweight= np.array([[node,centrality[node]] for node in centrality.keys()])
    maxindex = np.argsort(-nodeweight[:,1])[0:N]
    maxNode = nodeweight[maxindex,0]  
    
    return maxNode

def Identify_Sentinel(path,G,centrality,n):
    
    ReducedNodes = cg.ReducedNodeMap(path)
    topn = identifyTopN(centrality,n)
    seed = LocateSentinel(G,ReducedNodes,topn)#determine the seed on the basis of degree in block nodes 
    
    return seed
 
def RunSS(networkpath,resultpath):
 
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)    
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])
    sclique = [2,3,4,5,6,7,8,9]
    n = 10
    
    NRSSseeds={}
    for clique in sclique:
        print('clique',clique)
        edgelist = cg.load(networkpath+'/'+str(netIndex[clique])+'_reducedEdgelist')
        G_reducted = cg.weightNetwork(np.array(edgelist))
        Nodeweight = cg.NodeWeightDegree(G_reducted)
        NRSSseeds[clique] = Identify_Sentinel(networkpath,G,Nodeweight,n)
        
    Save(NRSSseeds,resultpath,'NRSSseeds')
   
    simulation = 1000
    alphas = np.arange(0,1.01,0.01)
    mu = 1

    NRSS_rhos={}
    for clique in NRSSseeds.keys():
        file = 'NRSS_' + str(clique) + '_spread'
        sentinels = NRSSseeds[clique]
        sentinel_avg_arho = Aspreadmseed(G,alphas,mu,sentinels,simulation,resultpath,file)   
        NRSS_rhos[clique] = sentinel_avg_arho
    
    Save(NRSS_rhos,resultpath,'NRSS_Spread')
 
def ProbInfectRG(G_reducted,infecteds,alpha):
    
    simutimes = 1000
    mu = 1
    Ninfected = {node: 0 for node in G_reducted.nodes()}
    for itr in np.arange(simutimes):
        [S,I,R,Allinfecteds] = cg.SIR_WG(G_reducted,alpha,mu,infecteds)
        for infected in Allinfecteds:
            Ninfected[infected]+=1
            
        for node in Ninfected.keys():
            Ninfected[node] = Ninfected[node]/simutimes             
        
        return Ninfected
    
def ProbInfectBeta(args):
    
    G_reducted,alphas,infecteds = args
    ProbInfBeat = {}    
    for alpha in alphas:
        ProbInfBeat[alpha] = ProbInfectRG(G_reducted,infecteds,alpha)
    
    return ProbInfBeat

def RunProbInfect(networkpath,resultpath):
    
    clique = 9
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)   
    edgelist = cg.load(networkpath+'/'+str(netIndex[clique])+'_reducedEdgelist')
    G_reducted = cg.weightNetwork(np.array(edgelist))
    
    seeds = set(G_reducted.nodes())
    alphas = np.arange(0,1,0.01)
    
    Nseed = 100
    args = []
    for itr in np.arange(Nseed):
        infected = random.choice(list(seeds))
        args.append([G_reducted,alphas,infected])
    
    pool = Pool(processes = 20)
    results = pool.map(ProbInfectBeta,args) 
    pk.dump(results,open(resultpath +'/InfectedProb','wb'))
    
    Nodeweight = cg.NodeWeightDegree(G_reducted)
    topn = identifyTopN(Nodeweight,5)
    nodestrength = {top:Nodeweight[topn[i]] for i,top in enumerate(np.arange(1,6))}
    node_infProb_avg = {}
    node_infProb_std = {}

    for i in np.arange(1,6):
        node = nodestrength[i]
        rho_beta = np.zeros((len(alphas),Nseed))
        for j,each in enumerate(results):
            for i,alpha in enumerate(alphas):
                rhos_dict = each[alpha]
                rho_beta[i,j] = rhos_dict[node]
        avg = np.average(rho_beta,axis = 1)
        std = np.std(rho_beta,axis = 1)
        node_infProb_avg[i] = avg
        node_infProb_std[i] = std
        
    pk.dump(node_infProb_avg,open(resultpath +'/node_infProb_avg','wb'))
    pk.dump(node_infProb_std,open(resultpath +'/node_infProb_std','wb'))
    pk.dump(nodestrength,open(resultpath +'/nodestrength','wb'))
    
def PlotAxes(ax,xlabel,ylabel,title='',fontsize=20, n_legend = 18, mode=True):
    
    font_label = {'family': "Calibri", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)
    if mode == True:
        ax.legend(loc='best',framealpha=0, fontsize=n_legend)
            
def PlotSSDiss(networkpath,resultpath,figurepath):
    
    colors = plt.get_cmap('Paired')
    n_legend = 18

    sclique = [2,3,4,5,6,7,8,9]
    beta = np.arange(0,1.01,0.01)
    markers = ['o','^','s','*','h']
    mec_color = 'black'#'#978C86'#'white'
    n_legend = 18
    markersize = 10
    mew = 0.5
    
    fig,ax = plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
    axes = ax[0].inset_axes([0.3,0.2,0.35,0.35])
    
    firstTime = pd.DataFrame()
    firstTime['beta'] = np.arange(0,1.01,0.01)
    for i,clique in enumerate(sclique):
        rhos = np.loadtxt(resultpath+'/'+'NRSS_'+str(clique)+'_spread.csv')
        firstTime[str(clique)] = rhos
        sns.scatterplot(data=firstTime,x='beta',y=str(clique),ax=ax[0],marker='o',color=colors(i),label=r'$NR_s('+str(clique+1)+')$',sizes=30)#,,label = file
        axes.plot(beta[0:31],rhos[0:31], marker='o',color=colors(i), ls='', mec='white')
        
    cg.PlotAxes(ax[0],r'$\beta$',r'$T_{fd}$','a')
    ax[0].legend(loc='best',framealpha=0, fontsize=n_legend)    
    PlotAxes(axes,r'$\beta$',r'$T_{fd}$','',fontsize=16, n_legend = 10)
    
    nodestrength = cg.load(resultpath +'/nodestrength')
    InfProb_Avg = cg.load(resultpath+'/node_infProb_avg')
    for i in np.arange(5):
        ax[1].plot(beta[:31],InfProb_Avg[i][:31],color=colors(i), ms=markersize,marker =markers[i], mec =mec_color,mew = mew, label = r'$K_w=$'+str(int(nodestrength[i+1])))
    cg.PlotAxes(ax[1],r'$\beta$',r'$P_{inf}$','b')
    ax[1].legend(loc='best',framealpha=0, fontsize=n_legend)    

    clique = 9
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)
    Oedgelist = cg.load(networkpath+'/0_reducedEdgelist')
    G = cg.weightNetwork(np.array(Oedgelist))
    edgelist = cg.load(networkpath+'/'+str(netIndex[clique])+'_reducedEdgelist')
    G_reducted = cg.weightNetwork(np.array(edgelist))
    Nodeweight = cg.NodeWeightDegree(G_reducted)
    topn = identifyTopN(Nodeweight,5)
    cliquelabels = cg.ReducedNodeMap(networkpath)
    
    block_path = {}
    for each in topn:
        block_nodes = cg.TrackBackNodes([each],cliquelabels) 
        blockG = nx.subgraph(G, block_nodes)
        block_path[each] = nx.average_shortest_path_length(blockG)
    
    tick_labels = ['$K_w=566$','$K_w=251$','$K_w=101$','$K_w=90$','$K_w=60$']
    x = np.arange(len(block_path))
    width = 0.35
    mec_color = 'white'
    for i,name in enumerate(block_path.keys()):
        ax[2].bar(x[i],block_path[name],width,color=colors(i),edgecolor = mec_color, linewidth=mew)
        
    cg.PlotAxes(ax[2],r'reducted node block',r'$\left\langle  d_{block} \right\rangle$','c')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(tick_labels,fontdict = {'family': "Arial", 'size':24}, rotation=35)

    plt.savefig(figurepath + '/FigS10.pdf')
    plt.savefig(figurepath + '/FigS10.eps')
    plt.savefig(figurepath + '/FigS10.png', dpi=600)

    
if __name__ == '__main__':    
    
    networkpath = root_path +'/NetworkReduction/fig3_reductionSize/network/GrQc'
    resultpath = root_path + '/NetworkReduction/sfig10_sentinel/result'
    figurepath = root_path + '/NetworkReduction/sfig10_sentinel/figure'
    
    #Application SS
    RunSS(networkpath,resultpath)
    RunProbInfect(networkpath,resultpath)
    PlotSSDiss(networkpath,resultpath,figurepath)
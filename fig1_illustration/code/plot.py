# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 22:24:08 2021

@author: Leyang Xue
"""

root_path = 'F:/work/work5_reductionability' #please change the root_path if run the code 

import sys
sys.path.append(root_path+'/NetworkReduction') 
from utils import coarse_grain as cg
import os 
import random 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import pickle as pk
import copy

def load_position(path):
    '''
    load the space position of networks in 2d

    Parameters
    ----------
    path : str
        path.

    Returns
    -------
    pos : dict
        space position of nodes in the network.

    '''
    file = open(path+'/network.cyjs')
    data_json = json.loads(file.read())
    network_nodes = data_json['elements']['nodes']
    pos = {}
    for each in network_nodes:
       name = each['data']['name']
       pos[float(name)] = np.array([each['position']['x'],each['position']['y']])
    
    return pos
    
def node_location(center,rho,n):
    
   theta = np.linspace(0,2*np.pi,n+1)
   locations = []
   for i in np.arange(n):
       locations.append(np.array([center[0]+rho*np.cos(theta[i]),center[1]+rho*np.sin(theta[i])]))

   return locations

def clique_layout(G,pos,rho):
    
    pos_clique = {}
    cliques = list(nx.find_cliques(G))
    for clique in cliques:
        if len(clique) > 2:
            clique_degree = np.array([[item[0],item[1]] for item in degree(clique)])
            nodes = clique_degree[np.argmax(clique_degree[:,1]),0]
            center = pos[nodes]
            locations = node_location(center,rho,len(clique)) 
            for i,clique_nodes in enumerate(clique):
                pos_clique[clique_nodes] = locations[i] 
        else:
            for i,clique_nodes in enumerate(clique):
                pos_clique[clique_nodes] = pos[clique_nodes]
    
    return pos_clique

def node_color(G,sub_nodes,color1,color2):
    
    #return the color of nodes
    
    subnetwork_nodecolor = []
    subnetwork_nodelist = []
    for each in G.nodes():
        if each in sub_nodes:
            subnetwork_nodelist.append(each)
            subnetwork_nodecolor.append(color1)
        else:
            subnetwork_nodelist.append(each)
            subnetwork_nodecolor.append(color2)
    
    return np.array(subnetwork_nodelist), subnetwork_nodecolor

def node_size(count,basic_size):
    
    #return the size of nodes
    
    nodelist = []
    nodesize = []
    for each in count.items():
        nodelist.append(each[0])
        nodesize.append(each[1]*basic_size)
    
    return np.array(nodelist), nodesize

def edge_size(Ew,width):
    
    #return the edgewidth of edges
    
    edgelist = []
    edgewidth = []
    for each in Ew.items():
        edgelist.append(each[0])
        edgewidth.append((each[1]-1)+width)
    
    return edgelist, edgewidth
        
def coarseGrain(path,filename,Rorder=3):
    '''
    Coarse graining the network until it is reduced into Rorder (3)-clique CGNS

    Parameters
    ----------
    path : str
        path to save the result.
    filename : str
        name.
    Rorder : int, optional
        k-clique CGNS. The default is 3.

    Returns
    -------
    Nnodes : array
        number of nodes over time.
    Nedges : array
        number of edges over time.
    sizeCliques : array
        size of clique over time.
    '''
    
    #load the original network
    edgelist = np.loadtxt(path+'/'+filename)
    G = cg.load_network(edgelist)
    edgelist = np.array([[edge[0],edge[1]] for edge in nx.to_edgelist(G)])
    [Nnodes, Nedges, sizeCliques]= cg.coarseGrain(edgelist,Rorder,path)
    
    return Nnodes, Nedges, sizeCliques

def loadRG(path,netIndex,sizeclique):
    '''
    load the k-clique CGNs

    Parameters
    ----------
    path : str
        path to save the CGNs.
    netIndex : dict
        map relation of name between supernodes and nodes.
    sizeclique : list
        size of k-clique over time.

    Returns
    -------
    RG : list
        CGNs.

    '''
    RG = {}
    for sc in sizeclique:
        reduce_data = cg.load(path+'/'+str(netIndex[sc])+'_reducedEdgelist')
        reduce_edgelist = np.array(reduce_data)[:,0:2]
        RG[sc+1] = cg.load_network(reduce_edgelist)
    
    return RG

def findMaxWeiNode(path,netIndex,sc,rank):
    '''
    return the top-rank node according to node weight in k(sc)-clique CGNs 
    note that top-n refers to the rank
    
    Parameters
    ----------
    path : str
        path to save the node weight of k-clique CGNs.
    netIndex : dict
        map relation of name between supernodes and nodes.
    sc : int
        size of k-clique.
    rank : int
        value of rank.

    Returns
    -------
    node : int 
        name of nodes.

    '''
    Nodeweight = cg.load(path+'/'+str(netIndex[sc])+'_Wnewnodes')
    nodew_array = np.array([[key,value] for key,value in Nodeweight.items()])
    weight_max = np.argsort(-nodew_array[:,1])[rank]
    node = nodew_array[weight_max,0]
    
    return node

def RGpos(subG,pos,cliquelabels):
    '''
    calculate the space position of subgraph

    Parameters
    ----------
    subG : graph
        subgraph in the original network.
    pos : dict
        space position in the original network.
    cliquelabels : dict
         the map relation of name between supernode and nodes.

    Returns
    -------
    pos_subG : TYPE
        DESCRIPTION.

    '''
    pos_subG ={}
    for each in subG.nodes():
        if each in pos.keys():
            pos_subG[each] = pos[each]
        else:
            ori_nodes = cg.TrackBackNodes([each], cliquelabels)
            nodes  = random.choice(ori_nodes)
            pos_subG[each] = pos[nodes]
            
    return pos_subG

def Save(data,path,filename):
    '''
    save the data as a format of pick 

    Parameters
    ----------
    data : any
        data.
    path : str
        path to save the data.
    filename : str
        filename.

    Returns
    -------
    None.

    '''
    pk.dump(data,open(path+'/'+filename,'wb'))

def Rnode_color(nodelist,nodesize,color1,color2):
    
    maxinx = np.argmax(np.array(nodesize))
    nodecolor = [color2 for x in nodelist]
    nodecolor[maxinx] = color1
    
    return nodecolor
        
def PlotSubG(ax,subG,subNw,subEw,subPos,i):
    '''
    Plot the subgraph

    Parameters
    ----------
    ax : axes
        axes.
    subG : graph
        subgraph.
    subNw : dict
        node weight.
    subEw : dict
        edge weight.
    subPos : dict
        position.
    i : int
        index.

    Returns
    -------
    None.

    '''
    basic_size = 200
    alpha = 1
    width = 1.0 
    #color1 = '#FFAEBC'
    color2 = '#efefef'
    
    mec_color='black'#'#000000'
    Roptions_node = {"node_shape":'o','linewidths':1.0, 'edgecolors':mec_color, "alpha":alpha}
    options_edge = {"edge_color":edge_color,"style":'solid', "alpha": alpha}
    
    [nodelist,nodesize] = node_size(subNw,basic_size)
    [edgelist,edgesize] = edge_size(subEw,width)
    
    subNw_labels = {each:subNw[each] for each in subNw.keys() if subNw[each]!= 1}
    subEw_labels = {(each[0],each[1]):subEw[each] for each in subEw.keys() if subEw[each] != 1}
    
    [subnode, subcolor]=  node_color(subG,subG.nodes(),color2,color2)
    nx.draw_networkx_nodes(subG, pos = subPos, nodelist=nodelist, node_size=nodesize, node_color=subcolor, ax=ax,**Roptions_node)
    nx.draw_networkx_edges(subG, pos = subPos, edgelist = edgelist, width = edgesize, ax=ax, **options_edge)

    nodelabels = nx.draw_networkx_labels(subG, pos = subPos, labels=subNw_labels,font_size=20, ax=ax,alpha=alpha)
    edgelabels = nx.draw_networkx_edge_labels(subG,pos = subPos, edge_labels=subEw_labels,font_size=15, ax=ax,alpha=alpha) 

if __name__ == '__main__':
    
    #set the network and subnetwork (k-clique CGNs) path  
    networkpath = root_path + '/NetworkReduction/fig1_illustration/network' 
    subnetworkpath =  root_path + '/NetworkReduction/fig1_illustration/subnetwork'
    figureparh = root_path + '/NetworkReduction/fig1_illustration/figure'
    os.chdir(networkpath)
    
    #1. load the network
    data = np.loadtxt('ca-netscience.mtx')
    G = cg.load_network(data)
    #calculate the degree of network
    degree =nx.degree(G)
    
    #2. coarse-graining the network
    filename ='ca-netscience.mtx'
    #please uncomment if want to re-run the coarse-graining process
    #[Nnodes, Nedges, sizeCliques] = coarseGrain(networkpath,filename) 
    sizeCliques = np.loadtxt(networkpath + '/sizeCliques.txt')    
    #return the timestep t and number of weighted edges fof k-clique CGNs
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)     
    
    #3 extract the subgraph from the original network so that we show the reduction process for the subgraph
    #set the value of k-clique 
    sc = 3
    rank = 1
    #return the map relation of name between supernode and nodes
    cliquelabels = cg.ReducedNodeMap(networkpath)
    #return the node with largest nodeweight
    maxnode = findMaxWeiNode(networkpath,netIndex,sc,rank)
    #track the nodes contained by supernode 
    subnodes = cg.TrackBackNodes([maxnode], cliquelabels)
    #return the subgraph 
    subG = nx.subgraph(G,subnodes)
    
    #4 load the space position for original network, CGNs, subgraph  
    #load the reducted network for R = 4,3,2
    sizeclique = [4,3,2]
    RG = loadRG(networkpath,netIndex,sizeclique)
    #load the space position of original network
    pos =load_position(networkpath)
    RG_pos ={}
    for name,rg in RG.items():
        #calculate the position of CGNs for k= 4,3,2
        RG_pos[name] = RGpos(rg,pos,cliquelabels) 
    #calculate the space position of subgraph
    subGpos = RGpos(subG,pos,cliquelabels) 
        
    #5 coarse-graining process of subgraph
    #please uncomment if want to re-run the coarse-graining process of subgraph
    #sub_data = np.array([[each[0],each[1]] for each in list(nx.to_edgelist(subG))]) 
    #np.savetxt(subnetworkpath+'/subedgelist.txt',sub_data,fmt = '%d')
    #[subNnodes, subNedges, subsizeCliques]= coarseGrain(subnetworkpath,'subedgelist.txt',Rorder=5)
    
    #6 load the node weight and edge weight for CGNs, subgraph
    #load the node weight and edge weight for CGNs of original networks
    Nw ={}
    Ew = {}
    for name,rg in RG.items():
        Nw[name] =  cg.load(networkpath+'/'+str(netIndex[name-1])+'_Wnewnodes')
        Ew[name] =  cg.load(networkpath+'/'+str(netIndex[name-1])+'_Wnewedges')
    #save the node weight and edgeweight for subgraph
    subNw = {}
    subEw = {}
    subRG ={}
    SNw = {node:1 for node in subG.nodes()}
    SEw = {edge:1 for edge in subG.edges()}
    Save(SNw,subnetworkpath,'0_Wnewnodes')
    Save(SEw,subnetworkpath,'0_Wnewedges')
    #load the node weight and edgeweight for subgraph
    for i in np.arange(0,5):
        subdata= cg.load(subnetworkpath+'/'+str(i)+'_reducedEdgelist')
        subedgelist = np.array(subdata)[:,0:2]
        subRG[i] = cg.load_network(subedgelist)
        subNw[i] = cg.load(subnetworkpath+'/'+str(i)+'_Wnewnodes')
        subEw[i] = cg.load(subnetworkpath+'/'+str(i)+'_Wnewedges')

    #7 calculate the space position of CGNs for subgraph
    subcliquelabels = cg.ReducedNodeMap(subnetworkpath)
    subRG_pos = {}
    for i in np.arange(0,5):
        subRG_pos[i] = RGpos(subRG[i],subGpos,subcliquelabels)

    #8 begin to plot the figure
    plt.figure(figsize=(32,18),tight_layout=True)
    ax1 = plt.subplot2grid((5,24), (2,0), colspan=6,rowspan=3)
    ax2 = plt.subplot2grid((5,24), (2,6), colspan=6,rowspan=3)
    ax3 = plt.subplot2grid((5,24), (2,12), colspan=6,rowspan=3)
    ax4 = plt.subplot2grid((5,24), (2,18), colspan=6,rowspan=3)        
    ax5 = plt.subplot2grid((5,24), (0,0), colspan=4,rowspan=2)
    ax6 = plt.subplot2grid((5,24), (0,4), colspan=4,rowspan=2)                 
    ax7 = plt.subplot2grid((5,24), (0,8), colspan=4,rowspan=2)                 
    ax8 = plt.subplot2grid((5,24), (0,12), colspan=4,rowspan=2)                 
    ax9 = plt.subplot2grid((5,24), (0,16), colspan=4,rowspan=2)                 
    ax10 = plt.subplot2grid((5,24), (0,20), colspan=4,rowspan=2)                 
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax7.axis('off')
    ax8.axis('off')
    ax9.axis('off')
    ax10.axis('off')
    
    title = ['','','','','','','','','','']
    font_label = {'family': "Calibri", 'size':30}

    ax1.set_title(title[0], fontdict=font_label)
    ax2.set_title(title[1], fontdict=font_label)
    ax3.set_title(title[2], fontdict=font_label)
    ax4.set_title(title[3], fontdict=font_label)
    ax5.set_title(title[4], fontdict=font_label)
    ax6.set_title(title[5], fontdict=font_label)
    ax7.set_title(title[6], fontdict=font_label)
    ax8.set_title(title[7], fontdict=font_label)
    ax9.set_title(title[8], fontdict=font_label)
    ax10.set_title(title[9], fontdict=font_label)
    
    ax1.set_xlim(-1130,1400)
    ax2.set_xlim(-1130,1400)
    ax3.set_xlim(-1130,1400)
    ax4.set_xlim(-1130,1400)

    ax5.set_xlim(-800,50)
    ax5.set_ylim(550,1780)
    ax6.set_xlim(-800,50)
    ax6.set_ylim(550,1780)
    ax7.set_xlim(-800,50)
    ax7.set_ylim(550,1780)
    ax8.set_xlim(-800,50)
    ax8.set_ylim(550,1780)
    ax9.set_xlim(-800,50)
    ax9.set_ylim(550,1780)
    ax10.set_xlim(-800,50)
    ax10.set_ylim(550,1780)
    
    nodes_color ='#efefef'   
    edge_color = '#5a5a5a' 
    color1 = '#FFAEBC'
    color2 = nodes_color
    
    mec_color='black'#'#000000'
    basic_size = 100
    width = 1
    alp = 1
    options_node = {"node_shape":'o','linewidths':1.0, 'edgecolors':mec_color,"node_size":basic_size, "alpha": alp}
    options_edge = {"edge_color":edge_color,"style":'solid', "alpha": alp}
    
    #options_labels = {"font_size":8, "font_color":"black","alpha":0.8}
    options_node_basic = {"node_shape":'o','linewidths':1.0, 'edgecolors':mec_color,"alpha": alp}
    reduce_options_node = {"node_color": nodes_color,"node_shape":'o','linewidths':1.0, 'edgecolors':mec_color, "alpha": alp}
    
    #8.1 extract the subG from the network 
    [subnode, subcolor]=  node_color(G,subRG[0].nodes(),color1,color2)
    nx.draw_networkx_nodes(G, pos,nodelist=subnode,node_color=subcolor,ax=ax1, **options_node)
    nx.draw_networkx_edges(G, pos, ax=ax1, **options_edge)
    #nx.draw_networkx_labels(G,pos, ax=ax[0],**options_labels)  
    
    #8.2 R = 8
    [r8nodelist,r8nodesize] = node_size(Nw[5],basic_size)
    [r8edgelist,r8edgesize] = edge_size(Ew[5],width)
    nx.draw_networkx_nodes(RG[5], pos = RG_pos[5], nodelist = r8nodelist, node_size = r8nodesize, ax=ax2, **reduce_options_node)
    nx.draw_networkx_edges(RG[5], pos = RG_pos[5], edgelist = r8edgelist, width = r8edgesize, ax=ax2, **options_edge)
    
    #8.3 R = 5
    [r5nodelist,r5nodesize] = node_size(Nw[4],basic_size)
    [r5edgelist,r5edgesize] = edge_size(Ew[4],width)
    nx.draw_networkx_nodes(RG[4], pos = RG_pos[4], nodelist=r5nodelist, node_size = r5nodesize, ax=ax3, **reduce_options_node)
    nx.draw_networkx_edges(RG[4], pos = RG_pos[4], edgelist = r5edgelist, width = r5edgesize, ax=ax3, **options_edge)
    
    #8.4 R = 3
    [r3nodelist,r3nodesize] = node_size(Nw[3],basic_size)
    [r3edgelist,r3edgesize] = edge_size(Ew[3],width)
    nx.draw_networkx_nodes(RG[3], pos = RG_pos[3], nodelist=r3nodelist, node_size = r3nodesize, ax=ax4, **reduce_options_node)
    nx.draw_networkx_edges(RG[3], pos = RG_pos[3], edgelist = r3edgelist, width = r3edgesize, ax=ax4, **options_edge)
    
    #8.5 sub original
    colores = '#FFBF86'
    PlotSubG(ax5,subRG[0],subNw[0],subEw[0],subRG_pos[0],i=0)

    #8.6 sub t=1
    colores = '#FFBF86'
    pos1 = copy.deepcopy(subRG_pos[1])
    pos1[352][0] =-550
    pos1[352][1] = 1300
    PlotSubG(ax6,subRG[1],subNw[1],subEw[1],pos1,i=1)
    
    #8.7 sub t=2
    colores = '#FFF47D'
    pos2 = copy.deepcopy(subRG_pos[2])
    pos2[353][0] =-350
    pos2[353][1] = 1450
    PlotSubG(ax7,subRG[2],subNw[2],subEw[2],pos2,i=2)
    
    #8.8 sub t =3  
    colores = '#FFF47D'
    pos3 = copy.deepcopy(subRG_pos[3])
    pos3[355.0][0] = -685
    pos3[355.0][1] = 760
    pos3[354.0][0] = -260
    pos3[354.0][1] = 1520
    PlotSubG(ax8,subRG[3],subNw[3],subEw[3],pos3,i=3)
    
    #8.9 sub t =4  
    PlotSubG(ax9,subRG[4],subNw[4],subEw[4],subRG_pos[4],i=4)
    
    #8.10 sub t = 5
    subRG[5] = nx.Graph()
    subRG[5].add_node(357)
    subNw[5] = {357.0:28}
    subEw[5] = {}
    subRG_pos[5]={357:[-550,1300]}
    PlotSubG(ax10,subRG[5],subNw[5],subEw[5],subRG_pos[5],i=5)
    
    #save the figure
    plt.savefig(figureparh+'/Figure.png', dpi=300)
    plt.savefig(figureparh+'/Figure.pdf')
    plt.savefig(figureparh+'/Figure.eps')


    
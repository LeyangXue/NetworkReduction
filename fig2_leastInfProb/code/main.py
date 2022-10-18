# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:55:13 2022

@author: Leyang Xue

"""
#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import numpy as np
import matplotlib.pyplot as plt
import sympy
import networkx as nx
import random 
import pickle as pk

def spreading_complete_G(n_order,simulation,betas,mu,path):
    '''
    Spreading on k-clique graph
    
    Parameters
    ----------
    n_order : int
        order of clique.
    simulation : int
        simulation times.
    betas : float
        infection rate.
    mu : float
        recover rate.
    path : str
        the path to save.

    Returns
    -------
    spreading : array
        the spreading at a given-order clique.

    '''
    complete_G = nx.complete_graph(n_order)
    node_set = list(complete_G.nodes())
    spreading  = np.zeros((len(betas),simulation))
    
    for j,each in enumerate(np.arange(simulation)):
        seed = random.choice(node_set)
        for i,beta in enumerate(betas):
            [S,I,R,Allinfecteds]= cg.SIR(complete_G,beta,mu,seed)
            spreading[i,j] = R[-1] 
            
    #pk.dump(spreading,open(path+'/'+str(n_order)+'_'+str(mu)+'_spreading','wb'))
    
    return spreading

def spreading_complete_G_N(n_order,n,simulation,betas,mu,path):
    '''
    Spreading with n initial seeds on k-clique graph

    Parameters
    ----------
    n_order : int
        order of clique.
    n: int
        the number of initial seed 
    simulation : int
        simulation times.
    betas : float
        infection rate.
    mu : float
        recover rate.
    path : str
        the path to save.

    Returns
    -------
    spreading : array
        the spreading at a given-order clique.

    '''
    if n < n_order:
        complete_G = nx.complete_graph(n_order)
        node_set = list(complete_G.nodes())
        spreading  = np.zeros((len(betas),simulation))
    
        for j,each in enumerate(np.arange(simulation)):
            seeds = random.sample(node_set,n)
            for i,beta in enumerate(betas):
                [S,I,R,Allinfecteds]= cg.SIR(complete_G,beta,mu,seeds)
                spreading[i,j] = R[-1] 
    else:
        print('n must smaller than n_order')
        
    #pk.dump(spreading,open(path+'/'+str(n_order)+'_'+str(n)+'_'+str(mu)+'_simulation','wb'))
    
    return spreading

def spreading_k_clique(largest_order,simulation,beta,mu,path):
    '''
    calculate the probability that one infected nodes occupy all nodes in k-clique graph
    
    Parameters
    ----------
    largest_order : int
        order of maximum clique.
    simulation : int
        simulation times.
    beta : float
        infection rate.
    mu : float
        recover rate.
    path : str
        path to save.

    Returns
    -------
    p_occupy_subnetwork : array
        the probability to infect all nodes in a clique at a given infection rate 

    '''
    p_occupy_subnetwork = np.zeros((len(beta),largest_order-1))
    
    for i,order in enumerate(np.arange(2, largest_order+1)):
        print('order number:', order)
        spreading = spreading_complete_G(order, simulation,beta, mu, path)
        spreading[spreading != order] = 0
        spreading[spreading == order] = 1 
        number_occupy_allnodes= np.sum(spreading,axis=1)
        p_occupy_all_nodes = number_occupy_allnodes /simulation
        p_occupy_subnetwork[:,i] = p_occupy_all_nodes
    
    pk.dump(p_occupy_subnetwork,open(path + '/'+str(largest_order)+'_'+str(mu)+'_occupy_probability','wb'))  
    
    return p_occupy_subnetwork

def spreading_k_clique_N(largest_order,n,simulation,beta,mu,path):
    '''
    calculate the probability that n infected nodes occupy all nodes in k-clique graph

    Parameters
    ----------
    largest_order : int
        order of maximum clique.
    n : int
        the number of initial seeds 
    simulation : int
        simulation times.
    beta : float
        infection rate.
    mu : float
        recover rate.
    path : str
        path to save.

    Returns
    -------
    p_occupy_subnetwork : array
        the probability to infect all nodes in a clique at a given infection rate 

    '''
    p_occupy_subnetwork = np.zeros((len(beta),largest_order-n))
    
    for i,order in enumerate(np.arange(n+1, largest_order+1)):
        print('order number:', order)
        
        spreading = spreading_complete_G_N(order,n,simulation,beta, mu, path)
        spreading[spreading != order] = 0
        spreading[spreading == order] = 1 
        number_occupy_allnodes= np.sum(spreading,axis=1)
        p_occupy_all_nodes =number_occupy_allnodes /simulation
        p_occupy_subnetwork[:,i] = p_occupy_all_nodes
    
    pk.dump(p_occupy_subnetwork,open(path + '/'+str(largest_order)+'_1_n'+str(n)+'_occupy_probability','wb'))  
    
    return p_occupy_subnetwork

def SpreadClique(path):
    '''
    For different recovery probability, 
    calculate the probability that one infected node infect all nodes in k-clique graph 

    Parameters
    ----------
    path : str
        path to save.

    Returns
    -------
    p_occupy10 : array
        probability with u=1.
    p_occupy02 : array
        probability with u=0.2.
    p_occupy05 : array
        probability with u=0.5.
    p_occupy08 : array
        probability with u=0.8.
    '''
    simulation = 100000
    largest_order = 20
    
    betas = np.arange(0,1.01,0.01)
    mu = 0.2
    p_occupy02 = spreading_k_clique(largest_order,simulation,betas,mu,path)
    mu = 0.5
    p_occupy05 = spreading_k_clique(largest_order,simulation,betas,mu,path)
    mu = 0.8
    p_occupy08 = spreading_k_clique(largest_order,simulation,betas,mu,path)
    mu = 1
    p_occupy10 = spreading_k_clique(largest_order,simulation,betas,mu,path)

    return p_occupy10,p_occupy02,p_occupy05,p_occupy08

def SpreadCliqueN(path):
    '''
    For mu=1
    calculate the probability that n infected node infect all nodes in k-clique graph 

    Parameters
    ----------
    path : str
        path to save the data.

    Returns
    -------
    None.

    '''
    simulation = 100000
    largest_order = 20
    
    betas = np.arange(0,1.01,0.01)
    mu = 1
    ns = [2,3,4]
    for n in ns:
        spreading_k_clique_N(largest_order,n,simulation,betas,mu,path)

def SpreadClique100(path):
    '''
    calculate the probability form 2 to 100

    Parameters
    ----------
    path : str
        path to save.

    Returns
    -------
    None.

    '''
    mu = 1
    betas = np.arange(0,1.01,0.01)
    largest_order = 100 
    simulation = 10000
    p_occupy10 = spreading_k_clique(largest_order,simulation,betas,mu,path)

def OccupyClique():
    '''
    Calculate the probability of infecting all nodes in a clique by symbol expression for beta \in (0,1)

    Returns
    -------
    theory_result : list
        each element in the list denotes symbol expression.

    '''
    beta = sympy.symbols('beta')
    
    s = 1-beta
    
    y2 = beta
    #f2 = sympy.expand(y2)
    
    y3 = np.power(beta,2) + 2*np.power(beta,1)*s*y2 #np.power(beta,2)*(3-2*beta)
    #f3 = sympy.expand(y3) 
    
    y21_4 = 1-np.power(s,2)
    y4 = np.power(beta,3) + 3*np.power(beta,2)*s*y21_4 + 3*np.power(beta,1)*np.power(s,2)*y3
    #f4 = sympy.expand(y4)
    
    y31_5 = 1-np.power(s,3)
    y22_5=  1-(np.power(s,4) + 2*y21_4*np.power(s,3)) #(4*np.power(beta,2)+2*np.power(beta,3))*np.power(s,2)+4*np.power(beta,2)*np.power(s,3)+4*np.power(beta,3)*s+np.power(beta,4)
    y5 = np.power(beta,4) + 4*np.power(beta,3)*s*y31_5 + 6*np.power(beta,2)*np.power(s,2)*y22_5 + 4*beta*np.power(s,3)*y4
    #f5= sympy.expand(y5) 
    
    y41_6 = 1-np.power(s,4)
    y32_6 = 1-(np.power(s,6) + 2*y31_5*np.power(s,4))# 6*np.power(s,7)*beta+3*np.power(s,6)*np.power(beta,2)+6*np.power(s,6)*np.power(beta,2)+12*np.power(s,5)*np.power(beta,3)+3*np.power(beta,4)*np.power(s,4)+6*np.power(beta,2)*np.power(s,6)+12*np.power(beta,2)*np.power(s,7)
    y23_6 = 1-(np.power(s,6) + 3*y21_4*np.power(s,6) + 3*y22_5*np.power(s,4))#6*np.power(s,7)*beta+3*np.power(s,6)*np.power(beta,2)+6*np.power(s,6)*np.power(beta,2)+12*np.power(s,5)*np.power(beta,3)+3*np.power(beta,4)*np.power(s,4)+6*np.power(beta,2)*np.power(s,6)+12*np.power(beta,2)*np.power(s,7)
    y6 = np.power(beta,5) + 5*np.power(beta,4)*s*y41_6 +10*np.power(beta,3)*np.power(s,2)*y32_6 + 10*np.power(beta,2)*np.power(s,3)*y23_6 + 5*beta*np.power(s,4)*y5
    #f6 = sympy.expand(y6)
    
    y51_7 = 1-np.power(s,5)
    y42_7 = 1-(np.power(s,8) + 2*y41_6*np.power(s,5)) #8*beta*np.power(s,8)+12*np.power(beta,2)*np.power(s,7)+8*np.power(beta,3)*np.power(s,6)+2*np.power(beta,4)*np.power(s,5)
    y33_7 = 1-(np.power(s,9) + 3*y31_5*np.power(s,8) + 3*y32_6*np.power(s,5))
    y24_7 = 1-(np.power(s,8) + 4*y21_4*np.power(s,9) + 6*y22_5*np.power(s,8) + 4*y23_6*np.power(s,5))
    y7 = np.power(beta,6)+6*beta*np.power(s,5)*y6+15*np.power(beta,2)*np.power(s,4)*y24_7 + 20*np.power(beta,3)*np.power(s,3)*y33_7 + 15*np.power(beta,4)*np.power(s,2)*y42_7+6*np.power(beta,5)*s*y51_7
    #f7 = sympy.expand(y7)
    
    y61_8 = 1-np.power(s,6)
    y52_8 = 1 - (np.power(s,10) + 2*y51_7*np.power(s,6))
    y43_8 = 1 - (np.power(s,12) + 3*y41_6*np.power(s,10) + 3*np.power(s,6)*y42_7)
    y34_8 = 1 - (np.power(s,12) + 4*y31_5*np.power(s,12) + 6*y32_6*np.power(s,10) + 4*y33_7*np.power(s,6))
    y25_8 = 1 - (np.power(s,10) + 5*y21_4*np.power(s,12) + 10*y22_5*np.power(s,12) + 10*y23_6*np.power(s,10) + 5*y24_7*np.power(s,6))
    y8 = np.power(beta,7) + 7*np.power(beta,6)*s*y61_8 +21*np.power(beta,5)*np.power(s,2)*y52_8 + 35*np.power(beta,4)*np.power(s,3)*y43_8+35*np.power(beta,3)*np.power(s,4)*y34_8+21*np.power(beta,2)*np.power(s,5)*y25_8 + 7*beta*np.power(s,6)*y7
    #f8 = sympy.expand(y8)
    
    y71_9 = 1-np.power(s,7)
    y62_9 = 1-(np.power(s,12) + 2*y61_8*np.power(s,7))
    y53_9 = 1-(np.power(s,15) + 3*y51_7*np.power(s,12) + 3*y52_8*np.power(s,7))
    y44_9 = 1-(np.power(s,16) + 4*y41_6*np.power(s,15) + 6*y42_7*np.power(s,12) + 4*y43_8*np.power(s,7))
    y35_9 = 1-(np.power(s,15) + 5*y31_5*np.power(s,16) + 10*y32_6*np.power(s,15) + 10*y33_7*np.power(s,12)+5*y34_8*np.power(s,7))
    y26_9 = 1-(np.power(s,12) + 6*y21_4*np.power(s,15) + 15*y22_5*np.power(s,16) + 20*y23_6*np.power(s,15)+15*y24_7*np.power(s,12)+6*y25_8*np.power(s,7))
    y9 = np.power(beta,8) + 8*np.power(beta,7)*s*y71_9 + 28*np.power(beta,6)*np.power(s,2)*y62_9 + 56*np.power(beta,5)*np.power(s,3)*y53_9 + 70*np.power(beta,4)*np.power(s,4)*y44_9+56*np.power(beta,3)*np.power(s,5)*y35_9 + 28*np.power(beta,2)*np.power(s,6)*y26_9 + 8*beta*np.power(s,7)*y8
    #f9 = sympy.expand(y9)
    
    y81_10 = 1 - np.power(s,8)
    y72_10 = 1 - (np.power(s,14) + 2*y71_9*np.power(s,8))
    y63_10 = 1 - (np.power(s,18) + 3*y61_8*np.power(s,14) + 3*y62_9*np.power(s,8))
    y54_10 = 1 - (np.power(s,20) + 4*y51_7*np.power(s,18) + 6*y52_8*np.power(s,14) + 4*y53_9*np.power(s,8))
    y45_10 = 1 - (np.power(s,20) + 5*y41_6*np.power(s,20) + 10*y42_7*np.power(s,18) + 10*y43_8*np.power(s,14) + 5*y44_9*np.power(s,8))
    y36_10 = 1 - (np.power(s,18) + 6*y31_5*np.power(s,20) + 15*y32_6*np.power(s,20) + 20*y33_7*np.power(s,18) + 15*y34_8*np.power(s,14) + 6*y35_9*np.power(s,8))
    y27_10 = 1 - (np.power(s,14) + 7*y21_4*np.power(s,18) + 21*y22_5*np.power(s,20) + 35*y23_6*np.power(s,20) + 35*y24_7*np.power(s,18) + 21*y25_8*np.power(s,14) +7*y26_9*np.power(s,8))
    y10 = np.power(beta,9) + 9*np.power(beta,8)*s*y81_10 + 36*np.power(beta,7)*np.power(s,2)*y72_10 + 84*np.power(beta,6)*np.power(s,3)*y63_10 + 126*np.power(beta,5)*np.power(s,4)*y54_10 + 126*np.power(beta,4)*np.power(s,5)*y45_10 + 84*np.power(beta,3)*np.power(s,6)*y36_10 + 36*np.power(beta,2)*np.power(s,7)*y27_10 + 9*beta*np.power(s,8)*y9
    #f10 = sympy.expand(y10)
    
    theory_result = [y2,y3,y4,y5,y6,y7,y8,y9,y10]
    
    return theory_result

def TheoryResult():
    '''
    The analytical probability to infect all node in a clique by a infected node
    
    Returns
    -------
    theoriticalValue : array
        the probability of infecting all nodes in a clique when a node is infected.
    '''
    TheoryExpress = OccupyClique()
    betas = np.arange(0,1.01,0.01)
    theoriticalValue = np.zeros((len(betas),len(TheoryExpress)))
    beta = sympy.symbols('beta')
    for j, f in enumerate(TheoryExpress):
        for i,alpha in enumerate(betas):
            alpha = round(alpha,2)
            theoriticalValue[i,j] = f.subs(beta,alpha)
    
    return  theoriticalValue

def InfectCriticalCondition(theoritical_result):
    '''
    calculate the least infection probability (\hat{beta_k}) required for occupying all nodes in the k-clique by one infected node

    Parameters
    ----------
    theoritical_result : array
        analytical results

    Returns
    -------
    sizeClique : dict
        \hat{beta_k}.

    '''
    sizeClique ={}
    beta = np.arange(0,1.01,0.01)
    for i in np.arange(theoritical_result.shape[1]):
        cr = beta[min(np.where(theoritical_result[:,i]>0.995)[0])]
        sizeClique[i+2] = round(cr,2)
    
    return sizeClique

def InfectCriticalCondition_N(theoritical_result,n):
    '''
    calculate the least infection probability (\hat{beta_k}) required for occupying all nodes in the k-clique by n infected node

    Parameters
    ----------
    theoritical_result : array
        analytical results.
    n : int
        number of initial seeds.

    Returns
    -------
    sizeClique : dict
        \hat{beta_k}, the least infected probability required for occupying all node by n initial infected node in k-clique .

    '''
    sizeClique ={}
    beta = np.arange(0,1.01,0.01)
    for i in np.arange(theoritical_result.shape[1]):
        cr = beta[min(np.where(theoritical_result[:,i]>0.995)[0])]
        sizeClique[i+n] = round(cr,2)
    
    return sizeClique

def node_color(G,sub_nodes,color1,color2):
    '''
    return the color of nodes 

    Parameters
    ----------
    G : graph
        network.
    sub_nodes : list
        a subset of node with color1.
    color1 : str
        color1.
    color2 : str
        color2.

    Returns
    -------
    subnetwork_nodelist : array
        node list.
    subnetwork_nodecolor : TYPE
        node color.

    '''
    subnetwork_nodecolor = []
    subnetwork_nodelist = []
    for each in G.nodes():
        if each in sub_nodes:
            subnetwork_nodelist.append(each)
            subnetwork_nodecolor.append(color1)
        else:
            subnetwork_nodelist.append(each)
            subnetwork_nodecolor.append(color2)
    
    subnetwork_nodelist = np.array(subnetwork_nodelist)
    
    return subnetwork_nodelist, subnetwork_nodecolor

def drawClique(n,ax):
    '''
    draw n-clique graph

    Parameters
    ----------
    n : int
        value of n-clique.
    ax : axes
        axes.

    Returns
    -------
    None.

    '''
    color = plt.get_cmap('Paired')
    basic_size = 200
    mec_color='black'
    edge_color = '#7f7f7f' 
    color1 = color(0) #'#B9E3FA'
    color2 = color(6) #'#FFCB78'

    completeG = nx.complete_graph(n)

    x1=-1.3
    x2= 1.3
    ax.set_xlim(x1,x2)
    ax.axis('off')
    y1 = -1.3
    y2 = 1.3
    ax.set_ylim(y1,y2)
    
    #figure
    options_node = {"node_shape":'o','linewidths':2.0, 'edgecolors':mec_color,"node_size":basic_size, "alpha": 1}
    options_edge = {"edge_color":edge_color,"style":'solid','width':2.0, "alpha":1}
    
    pos = nx.kamada_kawai_layout(completeG)
    sub_nodes = [min(completeG.nodes())]
    [nodelist,nodecolor]= node_color(completeG,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(completeG,pos = pos, nodelist=nodelist,node_color=nodecolor, ax=ax, **options_node)
    nx.draw_networkx_edges(completeG,pos = pos, ax=ax, **options_edge)

def PlotClique(ax):
    '''
    draw the clique structure with different size

    Parameters
    ----------
    ax : axes
        axes.

    Returns
    -------
    None.

    '''
    for i,n in enumerate(np.arange(2,11,1)):
        drawClique(n,ax[i])
        
    font_label = {'family': "Arial", 'size':30,'weight':'demibold'}
    ax[0].set_title('a', loc='left',fontdict=font_label)

def PlotOccupyProb(ax,path,theoritical_result):
    '''
    plot the probability to occupy all nodes in k-clique by a infected node

    Parameters
    ----------
    ax : axes
        axes.
    path : str
        path to load the data.
    theoritical_result : array
        analytical results.

    Returns
    -------
    None.

    '''
    simulation_result = cg.load(path + '/20_1_occupy_probability')
    ct = InfectCriticalCondition(theoritical_result)
    x= np.arange(0,101,4)
    colors = plt.get_cmap('Paired')
    mec_color = 'black'
    theo_color = '#A3A9AA'
    markersize = 10
    n_legend = 18
    mew = 0.8
    alp = 1
    
    beta = np.arange(0,1.01,0.01)
    for i in np.arange(theoritical_result.shape[1]): 
        ax.plot(beta[x],simulation_result[:,i][x], marker ='o', color=colors(i),ms= markersize, mew = mew, mec=mec_color,ls='',label=str(i+2)+'')
        if i == 8: 
            ax.plot(beta,theoritical_result[:,i],c=mec_color,label='T.',lw=1.5,alpha=alp)
        else:
            ax.plot(beta,theoritical_result[:,i],c=mec_color,lw=1.5,alpha=alp)  
    for i in [0,3,6]:
        ax.vlines(ct[i+2],ymin=0, ymax=1, color = colors(i), linewidth=3)
    
    ax.text(ct[2]-0.14,0.2,r'$\hat{\beta}_2$='+str(ct[2]), color=colors(0),fontsize ='large')#bbox=dict(facecolor=colors[0],alpha=0.5)
    ax.text(ct[5]-0.16,0.2,r'$\hat{\beta}_5$='+str(ct[5]), color=colors(3),fontsize ='large')#bbox=dict(facecolor=colors[3],alpha=0.5)
    ax.text(ct[8]-0.16,0.2,r'$\hat{\beta}_8$='+str(ct[8]), color=colors(6),fontsize ='large')#bbox=dict(facecolor=colors[6],alpha=0.5)

    ax.legend(bbox_to_anchor=(0.0,1.00),loc='upper left',fontsize=n_legend,handlelength=1.5,labelspacing=0.4,framealpha=0,handletextpad=0.00)
    cg.PlotAxes(ax,r'$\beta$',r'$\Lambda^k_{1\rightarrow(k-1)}$','b')

def PlotBetaT(ax,path):
    '''
    Plot the \hat{beta_k} as a function of k-clique
    
    Parameters
    ----------
    ax : axes
        axes.
    path : str
        path to load the data.

    Returns
    -------
    None.

    '''
    simulation_result_10 = cg.load(path + '/20_1_occupy_probability')
    simulation_result_02 = cg.load(path + '/20_0.2_occupy_probability')
    simulation_result_05 = cg.load(path + '/20_0.5_occupy_probability')
    simulation_result_08 = cg.load(path + '/20_0.8_occupy_probability')

    ct10 = InfectCriticalCondition(simulation_result_10)
    ct02 = InfectCriticalCondition(simulation_result_02)
    ct05 = InfectCriticalCondition(simulation_result_05)
    ct08 = InfectCriticalCondition(simulation_result_08)

    marker_size = 10
    alp = 1
    mec_color = 'black'
    colors = plt.get_cmap('Paired')
    mew = 0.8
    
    ax.plot(ct02.keys(),ct02.values(),'o--',color =colors(1), alpha=alp, ms = marker_size, mec = mec_color, mew = mew, label = r'$\mu$=0.2')
    ax.plot(ct05.keys(),ct05.values(),'^--',color =colors(3), alpha=alp, ms = marker_size, mec = mec_color, mew = mew, label = r'$\mu$=0.5')
    ax.plot(ct08.keys(),ct08.values(),'s--',color =colors(5), alpha=alp, ms = marker_size, mec = mec_color, mew = mew, label = r'$\mu$=0.8')
    ax.plot(ct10.keys(),ct10.values(),'*--',color =colors(7), alpha=alp,ms = marker_size, mec = mec_color,  mew = mew, label = r'$\mu$=1.0')
    ax.set_xticks(np.arange(2,21,3))
    ax.set_xticklabels(np.arange(2,21,3))
    cg.PlotAxes(ax,'k-clique',r'$\hat{\beta}_k$','c')     
    ax.legend(loc='best',fontsize=18,handlelength=1.5,labelspacing=0.4,framealpha=0,handletextpad=0.00)

def PlotBetaTM(ax,path):
    '''
    Plot the \hat{beta_k} as a function of k-clique for n intial seeds 

    Parameters
    ----------
    ax : axes
        axes.
    path : str
        path to load the data.

    Returns
    -------
    None.

    '''
    simulation_result_1k = cg.load(path + '/20_1_occupy_probability')
    simulation_result_2k = cg.load(path + '/20_1_n2_occupy_probability')
    simulation_result_3k = cg.load(path + '/20_1_n3_occupy_probability')
    simulation_result_4k = cg.load(path + '/20_1_n4_occupy_probability')
    
    ct1 = InfectCriticalCondition_N(simulation_result_1k,2)
    ct2 = InfectCriticalCondition_N(simulation_result_2k,3)
    ct3 = InfectCriticalCondition_N(simulation_result_3k,4)
    ct4 = InfectCriticalCondition_N(simulation_result_4k,5)
    
    marker_size = 10
    alp = 1
    mew = 0.8
    mec_color = 'black'
    colors = plt.get_cmap('Paired')
    
    ax.plot(ct1.keys(),ct1.values(), '*--',color =colors(7), alpha=alp, mec = mec_color,ms = marker_size, mew = mew, label = r'$\Lambda^k_{1\rightarrow(k-1)}$')
    ax.plot(ct2.keys(),ct2.values(), 's--',color =colors(5), alpha=alp, mec = mec_color,ms = marker_size, mew = mew, label = r'$\Lambda^k_{2\rightarrow(k-2)}$')
    ax.plot(ct3.keys(),ct3.values(), '^--',color =colors(3), alpha=alp, mec = mec_color,ms = marker_size, mew = mew, label = r'$\Lambda^k_{3\rightarrow(k-3)}$')
    ax.plot(ct4.keys(),ct4.values(), 'o--',color =colors(1), alpha=alp, mec = mec_color,ms = marker_size, mew = mew, label = r'$\Lambda^k_{4\rightarrow(k-4)}$')

    cg.PlotAxes(ax,'k-clique',r'$\hat{\beta}_k$','d')
    ax.legend(loc='best',fontsize=18,handlelength=1.5,labelspacing=0.4,framealpha=0,handletextpad=0.00)

    
def PlotCliqueSpread(networkpath,figurepath):

    theoritical_result = TheoryResult()

    plt.figure(figsize=(20,8),tight_layout=True)
    ax1 = plt.subplot2grid((4,9), (1,0), colspan=3,rowspan=3)
    ax2 = plt.subplot2grid((4,9), (1,3), colspan=3,rowspan=3)
    ax3 = plt.subplot2grid((4,9), (1,6), colspan=3,rowspan=3)
    ax4 = plt.subplot2grid((4,9), (0,0), colspan=1,rowspan=1)
    ax5 = plt.subplot2grid((4,9), (0,1), colspan=1,rowspan=1)
    ax6 = plt.subplot2grid((4,9), (0,2), colspan=1,rowspan=1)
    ax7 = plt.subplot2grid((4,9), (0,3), colspan=1,rowspan=1)        
    ax8 = plt.subplot2grid((4,9), (0,4), colspan=1,rowspan=1)
    ax9 = plt.subplot2grid((4,9), (0,5), colspan=1,rowspan=1)                 
    ax10 = plt.subplot2grid((4,9), (0,6), colspan=1,rowspan=1)                 
    ax11 = plt.subplot2grid((4,9), (0,7), colspan=1,rowspan=1)                 
    ax12 = plt.subplot2grid((4,9), (0,8), colspan=1,rowspan=1)                 
    
    ax = [ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
    #draw the k-clique
    PlotClique(ax)
    #plot the probability to occupy all nodes by a infected node
    PlotOccupyProb(ax1,networkpath,theoritical_result)
    ax2.sharey(ax3)
    #plot the \hat{beta_k} as a function of k-clique
    PlotBetaT(ax2,networkpath)
    #plot the \hat{beta_k} as a function of k-clique for n initial seeds 
    PlotBetaTM(ax3,networkpath)
    #PlotKclique(ax3,path)
    
    plt.savefig(figurepath+'/fig2.png',dpi=600)
    plt.savefig(figurepath+'/fig2.eps')
    plt.savefig(figurepath+'/fig2.pdf')

if __name__ == '__main__':
    
    resultpath = root_path + '/NetworkReduction/fig2_leastInfProb/result' #network path 
    figurepath = root_path + '/NetworkReduction/fig2_leastInfProb/figure' #figure path 
    
    #1. calculate the probability that one infected node infect all nodes in k-clique graph
    [p10,p02,p05,p08]= SpreadClique(resultpath)
    #for mu=1, calculate the probability that one infected node infect all nodes in k-clique graph, 
    SpreadCliqueN(resultpath)
    #for k-clique = 100, calculate the probability that one infected node infect all nodes in k-clique graph, for mu=1
    SpreadClique100(resultpath)

    #2 plot the spread on completer G
    PlotCliqueSpread(resultpath,figurepath)
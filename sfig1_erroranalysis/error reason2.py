# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:20:24 2021

@author: Leyang Xue
"""

import sys
sys.path.append("F:/work/work5_reductionability/")
import networkx as nx
import numpy as np
from utils import network_reduction as nr
import matplotlib.pyplot as plt 
from sympy import *
import os

if __name__ == '__main__':
    
    path = 'F:/work/work5_reductionability/code_2-1/program2/error_analysis'
    os.chdir(path)
    os.getcwd()

    edge_list = np.array([['A','B'],['B','C'],['B','D'],['B','E'],['C','E'],['C','D'],['C','F'],['D','E'],['E','F']])
    G = nx.from_edgelist(edge_list)
    A = nx.to_numpy_array(G)
    edge_list_r = np.array([['A','G'],['G','F']])
    G1 = nx.from_edgelist(edge_list_r)
    clique = np.array([['B','C'],['B','D'],['B','E'],['C','E'],['C','D'],['D','E']])
    G_clique = nx.from_edgelist(clique)


    p = np.arange(0,1.01,0.01)
    infecteds = ['A']
    
    simulation = 100000
    count = 0
    result = np.zeros(len(p))
    for i,beta in enumerate(p):
        count = 0
        for j in np.arange(simulation):
            [S,I,R]= nr.SIR_G(G, beta, infecteds[0])
            if 'F' in time_series:
                 count = count + 1
        result[i] = count/simulation

    #p(a->all)
    beta = symbols('beta')
    u = np.power(beta,3)+3*np.power(beta,2)*(1-beta)*(1-np.power(1-beta,2))+np.power(beta,2)*(3-2*beta)*np.power(1-beta,2)*(2*beta+1)
    a= (1-np.power(1-beta,2))*u*beta
    f =2*np.power(beta,2)*(np.power(beta,2)*(3-2*beta)*np.power(1-beta,3)+np.power(1-beta,4)*beta)+a
    f_exp= expand(f)
    
    #y4
    beta = symbols('beta')
    s = 1-beta
    y2 = beta
    y3 = np.power(beta,2) + 2*np.power(beta,1)*s*y2 #np.power(beta,2)*(3-2*beta)
    y21_4 = 1-np.power(s,2)
    y4 = np.power(beta,3) + 3*np.power(beta,2)*s*y21_4 + 3*np.power(beta,1)*np.power(s,2)*y3
    f_4 = expand(y4) 
    
    G0_analytical = np.zeros(len(p))
    Gr_analytical = np.zeros(len(p))
    Gclique = np.zeros(len(p))
    for i,beta in enumerate(p):
        Gr_analytical[i] = 2*beta**2-beta**3#beta*(1-np.power((1-beta),2))
        Gclique[i] = -6*beta**6 + 24*beta**5 - 33*beta**4 + 16*beta**3
        G0_analytical[i] = 4*beta**9 - 19*beta**8 + 32*beta**7 - 19*beta**6 - 3*beta**5 + 4*beta**4 + 2*beta**3
    
    x_index = np.min(np.where(Gclique>0.995)[0])/100
    x_equal = np.where((Gr_analytical - G0_analytical)<0.005)[0][6]/100
    
    #pc=0.38
    n_legend = 10
    marker_size = 8
    lw=1.5
    markers =['s','o','^','<','>','v','h','p','x','d','D','H','1','2','3']
    font_label = {'family': "Calibri", 'size':20}
    mec_color = '#363333'#'#978C86'
    colors = plt.get_cmap('Set1') #["#ff4b70","#a3d55e","#7e009f","#ff9d58","#7b89ff","#705300","#ff4fd6","#009f9d","#e1003f","#004779"]
    
    fig,ax = plt.subplots(1,3,figsize=(18,6),tight_layout=True)
    x=np.arange(0,101,4)

    #plot the numericall and theoreticall results, and lambda^4_{1\rightarrow3}(\beta)
    ax[2].plot(p[x],G0_analytical[x],'s',label = r'$p(A \rightarrow F|G_o): 2\beta^3 +  4\beta^4 - 3\beta^5 - 19\beta^6  + 32\beta^7 - 19\beta^8+4\beta^9 $',markersize=marker_size,mec='white',color=colors(1))
    ax[2].plot(p[x],result[x],'-',label = r'$p(A \rightarrow F|G_o):$ numerical simulation',lw=lw,mec=mec_color,color=mec_color)
    ax[2].plot(p[x],Gr_analytical[x],'^',label = r'$p(A \rightarrow F|G_r): 2\beta^2-\beta^3$',markersize=marker_size,mec='white',color=colors(2))
    ax[2].plot(p[x],Gclique[x],'o-',label = r'$\Lambda^4_{1 \rightarrow 3}(\beta)$: $-6\beta^6 + 24\beta^5 - 33\beta^4 + 16\beta^3$',markersize=marker_size,mec=mec_color,color=colors(0))
    
    ax[2].vlines(x_index,0,1,color=colors(8),linestyles='dashed')
    ax[2].vlines(x_equal,0,1,color=colors(8),linestyles='dashed')
    ax[2].annotate(r'$p(A \rightarrow F|G_o)=p(A \rightarrow F|G_r)$', xy=(x_equal, 0.25), xytext=(x_equal-0.35, 0.15),arrowprops=dict(arrowstyle="->"),color =mec_color)
    ax[2].annotate(r'$\hat{\beta}_4$', xy=(x_index, 0.25), xytext=(x_index+0.05, 0.15),arrowprops=dict(arrowstyle="->"),color =mec_color)

    ax[2].set_ylabel(r'$p$',fontdict = font_label)
    ax[2].set_xlabel(r'$\beta$',fontdict = font_label)
    ax[2].legend(loc='best',fontsize=n_legend,framealpha=0)
    ax[2].tick_params(direction='in', which='both',length =2, width=0.5, labelsize=n_legend)
    
    ax[0].set_title(r'(a) $G_o$',fontdict=font_label)
    ax[1].text(0.06,0.35,r'4-clique $\rightarrow$ Node:G',fontdict=font_label)
    ax[1].set_title(r'(b) $G_r$',fontdict=font_label)
    ax[2].set_title('(c)',fontdict=font_label)
    
    nx.draw(G,pos={'A':[0.9,0],'B':[0.6,0.3],'C':[0.4,0.3],'D':[0.4,0.5],'E':[0.6,0.5],'F':[0.1,0.8]},ax=ax[0],node_size=320,edge_color=mec_color,with_labels=True)
    nx.draw(G1,pos={'A':[0.9,0],'G':[0.5,0.4],'F':[0.1,0.8]},ax=ax[1],node_size=320,edge_color=mec_color,with_labels=True)
    nx.draw(G_clique,pos={'B':[0.3,0.1],'C':[0.1,0.1],'D':[0.1,0.3],'E':[0.3,0.3]},ax=ax[1],node_size=320,edge_color=mec_color,with_labels=True)
    
    plt.savefig('sfigure1.png')
    plt.savefig('sfigure1.eps')

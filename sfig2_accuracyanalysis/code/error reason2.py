# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:20:24 2021

@author: Leyang Xue
"""

#please change the current path if run the code
root_path  = 'G:/work/work5_reductionability'

import sys
sys.path.append("G:/work/work5_reductionability/")
import networkx as nx
import numpy as np
from utils import coarse_grain as cg
import matplotlib.pyplot as plt 
import pickle as pk
from sympy import symbols, expand
import os

if __name__ == '__main__':
    
    path = root_path +'/NetworkReduction/sfig2_accuracyanalysis'
    resultpath = root_path + '/NetworkReduction/sfig2_accuracyanalysis/result'
    figurepath = root_path + '/NetworkReduction/sfig2_accuracyanalysis/figure'
    loadpath = root_path + '/NetworkReduction/fig2_leastInfProb/result'
    
    result_beta = cg.load(loadpath+'/100_1_occupy_probability')
    result_beta4 = result_beta[:,2]
    
    os.chdir(path)
    os.getcwd()

    edge_list = np.array([['A','B'],['B','C'],['B','D'],['B','E'],['C','E'],['C','D'],['C','F'],['D','E'],['E','F']])
    G = nx.from_edgelist(edge_list)
    A = nx.to_numpy_array(G)
    edge_list_r = np.array([['A','H',1],['H','F',2]])
    G1 = cg.weightNetwork_example(edge_list_r)
    #G1 = nx.from_edgelist(edge_list_r)
    clique = np.array([['B','C'],['B','D'],['B','E'],['C','E'],['C','D'],['D','E']])
    G_clique = nx.from_edgelist(clique)

    p = np.arange(0,1.01,0.01)
    infecteds = ['A']
    
    #numerical simulation for G
    # simulation = 100000
    # count = 0
    # result = np.zeros(len(p))
    # for i,beta in enumerate(p):
    #     count = 0
    #     for j in np.arange(simulation):
    #         [S, I, R, t, time_series]= cg.SIR_G(G, beta,infecteds[0])
    #         if 'F' in time_series:
    #              count = count + 1
    #     result[i] = count/simulation
    # pk.dump(np.array(result),open(resultpath+'/spread.pkl','wb'))
    result = cg.load(resultpath+'/spread.pkl')
    
    #numerical simulation for G1
    # simulation = 100000
    # count = 0
    # result_G1 = np.zeros(len(p))
    # for i,beta in enumerate(p):
    #     count = 0
    #     for j in np.arange(simulation):
    #         [S, I, R, time_series]= cg.SIR_WG(G1, beta, 1, infecteds[0], tmin = 0, tmax = float('Inf'))
    #         if 'F' in time_series:
    #               count = count + 1
    #     result_G1[i] = count/simulation
    #pk.dump(np.array(result_G1),open(resultpath+'/spread_G1.pkl','wb'))  
    result_G1 = cg.load(resultpath+'/spread_G1.pkl')

    
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
    n_legend = 18
    marker_size = 10
    lw=2
    markers =['s','o','^','<','>','v','h','p','x','d','D','H','1','2','3']
    font_label = {'family': "Calibri", 'size':20}
    mec_color = '#363333'#'#978C86'
    colors = plt.get_cmap('Set1') #["#ff4b70","#a3d55e","#7e009f","#ff9d58","#7b89ff","#705300","#ff4fd6","#009f9d","#e1003f","#004779"]
    
    fig,ax = plt.subplots(1,3,figsize=(21,7),tight_layout=True)
    x=np.arange(0,101,4)

    #plot the numericall and theoreticall results, and lambda^4_{1\rightarrow3}(\beta)
    #label = r'$p(A \rightarrow F|G_o): 2\beta^3 +  4\beta^4 - 3\beta^5 - 19\beta^6  + 32\beta^7 - 19\beta^8+4\beta^9 $'
    ax[2].plot(p[x],G0_analytical[x],'--', lw=lw, color=colors(1),label = r'Theo. $p(A \rightarrow F|G)$')
    #label = r'$p(A \rightarrow F|G_o):$ numerical simulation'
    ax[2].plot(p[x], result[x], 's', markersize=marker_size, mec='white', color=colors(1), label = r'Simu. $p(A \rightarrow F|G)$')
    #label = r'$p(A \rightarrow F|G_r): 2\beta^2-\beta^3$'
    ax[2].plot(p[x],Gr_analytical[x],'--', lw=lw, color=colors(2),label = r'Theo. $p(A \rightarrow F|G_{R=4})$')
    ax[2].plot(p[x],result_G1[x],'^', markersize=marker_size, mec='white',color=colors(2),label = r'Simu. $p(A \rightarrow F|G_{R=4})$')
    #label = r'$\Lambda^4_{1 \rightarrow 3}(\beta)$: $-6\beta^6 + 24\beta^5 - 33\beta^4 + 16\beta^3$'
    ax[2].plot(p[x],Gclique[x],'--', lw=lw, color=colors(0), label = r'Theo. $\Lambda^4_{1 \rightarrow 3}(\beta)$')
    ax[2].plot(p[x],result_beta4[x],'o',markersize=marker_size,mec=mec_color,color=colors(0),label = r'Simu. $\Lambda^4_{1 \rightarrow 3}(\beta)$')

    ax[2].vlines(x_index,0,1,color=colors(8),linestyles='dashed')
    ax[2].vlines(x_equal,0,1,color=colors(8),linestyles='dashed')
    ax[2].annotate(r'$p(A \rightarrow F|G)=p(A \rightarrow F|G_{R=4})$', xy=(x_equal, 0.15), xytext=(x_equal-0.54, 0.00),arrowprops=dict(arrowstyle="->"),color =mec_color,size=14)
    ax[2].annotate(r'$\hat{\beta}_4$', xy=(x_index, 0.15), xytext=(x_index+0.05, 0.00),arrowprops=dict(arrowstyle="->"),color =mec_color,size=14)

    #ax[2].set_ylabel(r'$p$',fontdict = font_label)
    #ax[2].set_xlabel(r'$\beta$',fontdict = font_label)
    #ax[2].tick_params(direction='in', which='both',length =2, width=0.5, labelsize=n_legend)
    cg.PlotAxes(ax[0],'','','a           $G$')
    cg.PlotAxes(ax[1],'','','b        $G_{R=4}$')
    cg.PlotAxes(ax[2],r'$\beta$',r'$p$','c')
    ax[2].legend(loc='best',fontsize=n_legend,framealpha=0)

    #ax[0].set_title(r'(a) $G_o$',fontdict=font_label)
    #ax[1].text(0.06,0.35,r'4-clique $\rightarrow$ Node:G',fontdict=font_label)
    #ax[1].set_title(r'(b) $G_r$',fontdict=font_label)
    #ax[2].set_title('(c)',fontdict=font_label)
    
    #nx.draw(G,pos={'A':[0.9,0],'B':[0.6,0.3],'C':[0.4,0.3],'D':[0.4,0.5],'E':[0.6,0.5],'F':[0.1,0.8]},ax=ax[0],node_size=320,edge_color=mec_color,with_labels=True)
    #nx.draw(G1,pos={'A':[0.9,0],'G':[0.5,0.4],'F':[0.1,0.8]},ax=ax[1],node_size=320,edge_color=mec_color,with_labels=True)
    #nx.draw(G_clique,pos={'B':[0.3,0.1],'C':[0.1,0.1],'D':[0.1,0.3],'E':[0.3,0.3]},ax=ax[1],node_size=320,edge_color=mec_color,with_labels=True)
    
    #ax[2].annotate(r'$p(A \rightarrow F|G_o)=p(A \rightarrow F|G_r)$', xy=(0.38, 0.5), xytext=(0, 0.35),arrowprops=dict(arrowstyle="->"),color =mec_color)
    #ax[0].set_title(r'(a) $G$',fontdict=font_label)
    ax[1].text(0.06,0.35,r'4-clique $\rightarrow$ Node:H',fontdict=font_label)
    #ax[1].set_title(r'(b) $G_{R=4}$',fontdict=font_label)
    fontsize = 15
    nx.draw(G,pos={'A':[0.9,0],'B':[0.6,0.3],'C':[0.4,0.3],'D':[0.4,0.5],'E':[0.6,0.5],'F':[0.1,0.8]},ax=ax[0],node_size=500,edge_color=mec_color,with_labels=True,font_color='white',font_size=fontsize)
    nx.draw(G1,pos={'A':[0.9,0],'H':[0.5,0.4],'F':[0.1,0.8]},ax=ax[1],node_size=500,edge_color=mec_color,with_labels=True,font_color='white',font_size=fontsize)
    nx.draw(G_clique,pos={'B':[0.3,0.1],'C':[0.1,0.1],'D':[0.1,0.3],'E':[0.3,0.3]},ax=ax[1],node_size=500,edge_color=mec_color,with_labels=True,font_color='white',font_size=fontsize)
    
    
    plt.savefig(figurepath+'/FigS2.png', dpi=300)
    plt.savefig(figurepath+'/FigS2.eps')
    plt.savefig(figurepath+'/FigS2.pdf')

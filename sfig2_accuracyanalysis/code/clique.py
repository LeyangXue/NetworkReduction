# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:49:32 2021

@author: Administrator
"""

import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt
import os
from sympy import *

def node_color(G,sub_nodes,color1,color2):
    
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

if __name__ == '__main__':
    
    os.chdir(r'F:\work\work_5 reductionability\code_2-1\program\temp')
    
    complete_2 = nx.complete_graph(2)
    complete_3 = nx.complete_graph(3)
    complete_4 = nx.complete_graph(4)
    complete_5 = nx.complete_graph(5)
    complete_6 = nx.complete_graph(6)
    complete_7 = nx.complete_graph(7)
    complete_8 = nx.complete_graph(8)
    complete_9 = nx.complete_graph(9)
    complete_9 = nx.complete_graph(10)
    
    plt.figure(figsize=(30,4),tight_layout=True)
    ax1 = plt.subplot2grid((1,8), (0,0), colspan=1,rowspan=1)
    ax2 = plt.subplot2grid((1,8), (0,1), colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((1,8), (0,2), colspan=1,rowspan=1)
    ax4 = plt.subplot2grid((1,8), (0,3), colspan=1,rowspan=1)        
    ax5 = plt.subplot2grid((1,8), (0,4), colspan=1,rowspan=1)
    ax6 = plt.subplot2grid((1,8), (0,5), colspan=1,rowspan=1)                 
    ax7 = plt.subplot2grid((1,8), (0,6), colspan=1,rowspan=1)                 
    ax8 = plt.subplot2grid((1,8), (0,7), colspan=1,rowspan=1)                 

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax7.axis('off')
    ax8.axis('off')

         
    x1=-1.5
    x2= 1.5
    
    ax1.set_xlim(x1,x2)
    ax1.set_ylim(x1,x2)
    ax2.set_xlim(x1,x2)
    ax2.set_ylim(x1,x2)
    ax3.set_xlim(x1,x2)
    ax3.set_ylim(x1,x2)
    ax4.set_xlim(x1,x2)
    ax4.set_ylim(x1,x2)
    ax5.set_xlim(x1,x2)
    ax5.set_ylim(x1,x2)
    ax6.set_xlim(x1,x2)
    ax6.set_ylim(x1,x2)
    ax7.set_xlim(x1,x2)
    ax7.set_ylim(x1,x2)
    ax8.set_xlim(x1,x2)
    ax8.set_ylim(x1,x2)

    basic_size = 1000
    mec_color='black'
    edge_color = '#5a5a5a' 
    color1 = '#ff8f00'
    color2 = '#1565c0'
    
    #figure1    
    options_node = {"node_shape":'o','linewidths':4.0, 'edgecolors':mec_color,"node_size":basic_size, "alpha": 0.85}
    options_edge = {"edge_color":edge_color,"style":'solid','width':2.0, "alpha": 0.85}
    
    pos2 = nx.kamada_kawai_layout(complete_2)
    sub_nodes = [min(complete_2.nodes())]
    [nodelist,nodecolor]= node_color(complete_2,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_2,pos = pos2, nodelist=nodelist,node_color=nodecolor, ax=ax1, **options_node)
    nx.draw_networkx_edges(complete_2,pos = pos2, ax=ax1, **options_edge)

    pos3 = nx.kamada_kawai_layout(complete_3)
    sub_nodes = [min(complete_3.nodes())]
    [nodelist,nodecolor]= node_color(complete_3,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_3,pos = pos3, nodelist=nodelist,node_color=nodecolor, ax=ax2, **options_node)
    nx.draw_networkx_edges(complete_3,pos = pos3, ax=ax2, **options_edge)
    
    pos4 = nx.kamada_kawai_layout(complete_4)
    sub_nodes = [min(complete_4.nodes())]
    [nodelist,nodecolor]= node_color(complete_4,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_4,pos = pos4, nodelist=nodelist,node_color=nodecolor, ax=ax3, **options_node)
    nx.draw_networkx_edges(complete_4,pos = pos4, ax=ax3, **options_edge)

    pos5 = nx.kamada_kawai_layout(complete_5)
    sub_nodes = [min(complete_5.nodes())]
    [nodelist,nodecolor]= node_color(complete_5,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_5,pos = pos5, nodelist=nodelist,node_color=nodecolor, ax=ax4, **options_node)
    nx.draw_networkx_edges(complete_5,pos = pos5, ax=ax4, **options_edge)
    
    pos6 = nx.kamada_kawai_layout(complete_6)
    sub_nodes = [min(complete_6.nodes())]
    [nodelist,nodecolor]= node_color(complete_6,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_6,pos = pos6, nodelist=nodelist,node_color=nodecolor, ax=ax5, **options_node)
    nx.draw_networkx_edges(complete_6,pos = pos6, ax=ax5, **options_edge)

    pos7 = nx.kamada_kawai_layout(complete_7)
    sub_nodes = [min(complete_7.nodes())]
    [nodelist,nodecolor]= node_color(complete_7,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_7,pos = pos7, nodelist=nodelist,node_color=nodecolor, ax=ax6, **options_node)
    nx.draw_networkx_edges(complete_7,pos = pos7, ax=ax6, **options_edge)
 
    pos8 = nx.kamada_kawai_layout(complete_8)
    sub_nodes = [min(complete_8.nodes())]
    [nodelist,nodecolor]= node_color(complete_8,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_8,pos = pos8, nodelist=nodelist,node_color=nodecolor, ax=ax7, **options_node)
    nx.draw_networkx_edges(complete_8,pos = pos8, ax=ax7, **options_edge)

    pos9 = nx.kamada_kawai_layout(complete_9)
    sub_nodes = [min(complete_9.nodes())]
    [nodelist,nodecolor]= node_color(complete_9,sub_nodes,color1,color2)
    nx.draw_networkx_nodes(complete_9,pos = pos9, nodelist=nodelist,node_color=nodecolor, ax=ax8, **options_node)
    nx.draw_networkx_edges(complete_9,pos = pos9,  ax=ax8, **options_edge)
    
    title = ['(a) 2-clique','(b) 3-clique','(c) 4-clique','(d) 5-clique','(e) 6-clique','(f) 7-clique','(g) 8-clique','(h) 9-clique']
    font_label = {'family': "Calibri", 'size':30}
    ax1.set_title(title[0], fontdict=font_label)
    ax2.set_title(title[1], fontdict=font_label)
    ax3.set_title(title[2], fontdict=font_label)
    ax4.set_title(title[3], fontdict=font_label)
    ax5.set_title(title[4], fontdict=font_label)
    ax6.set_title(title[5], fontdict=font_label)
    ax7.set_title(title[6], fontdict=font_label)
    ax8.set_title(title[7], fontdict=font_label)
    
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:53:01 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

def SpreadOneNet(OG, alphas, netIndex, Sclique, simulations, networkpath, resultpath):
    '''
    run the numerical simulation on original, unweight and weighted k-clique CGNs

    Parameters
    ----------
    OG : graph
        original network.
    alphas : array
        infection rate.
    netIndex : dict
        map between reduced network and index.
    Sclique : int
        value of k-clique CGNs.
    simulations : int
        simulation times.
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the spreading results

    Returns
    -------
    Omc : array
        results of numerical simulation on original networks.
    Rmc : array
        results of numerical simulation on unweight reduced networks.
    RWmc : TYPE
        results of numerical simulation on weighted reduced networks.
    '''
    
    edgelist = cg.load(networkpath +'/' + str(netIndex[Sclique]) +'_reducedEdgelist')
    RG = cg.weightNetwork(edgelist)
    Wnodes = cg.load(networkpath +'/' + str(netIndex[Sclique]) +'_Wnewnodes')
  
    mu = 1  
    args = []
    for itr in np.arange(simulations):
        args.append([itr,RG,OG,Wnodes,alphas,mu])
    
    pool = Pool(processes = 15)
    results = pool.map(cg.run_SIRWG_simulation,args)
    [Omc,Rmc,RWmc] = cg.parseResult_SIRWG(results,resultpath,Sclique)
    
    return Omc, Rmc, RWmc

def SpreadSimulations(networkpath,resultpath,Scliques,simulations):
    '''
    Perform the numerical simulation on original, unweight and weighted k-clique CGNs. 

    Parameters
    ----------
    path : str
        path to load the network.
    Scliques : int
        value of (k-1)-clique.
    simulations : int
        simulation times.

    Returns
    -------
    None.
    '''
    
    #load the original network 
    OG_edgelist = cg.load(networkpath +'/'+'0_reducedEdgelist')
    OG = cg.load_network(OG_edgelist[:,0:2])
    
    #spreading on original network and weight network 
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex, netNweight]= cg.NetworkInx(sizeCliques,networkpath)
    alphas = np.arange(0,1.01,0.01)
    Omcs = np.zeros((len(alphas),len(Scliques)))
    Rmcs = np.zeros((len(alphas),len(Scliques)))
    RWmcs = np.zeros((len(alphas),len(Scliques)))
    for i,sclique in enumerate(Scliques):
        print('sclique:',sclique)
        [Omc, Rmc, RWmc] = SpreadOneNet(OG,alphas,netIndex,sclique,simulations,networkpath,resultpath)
        Omcs[:,i] = Omc
        Rmcs[:,i] = Rmc
        RWmcs[:,i] = RWmc
        
    pk.dump(Omcs,open(resultpath +'/Omcs','wb'))
    pk.dump(Rmcs,open(resultpath + '/Rmcs','wb'))
    pk.dump(RWmcs,open(resultpath + '/Rwmcs','wb'))
     
# def SpreadProb(networkpath,resultpath,Scliques,simulations):
#     '''
#     For a given network, calculate the probability of nodes being infected in the original and reduced networks
#     and further quantify their difference using the Euclidean distance 
    

#     Parameters
#     ----------
#     networkpath : str
#         path to load the reduced network.
#     resultpath : str
#         path to save the result.
#     Scliques : float
#         value of k-clique.
#     simulations : float
#         simulation times.

#     Returns
#     -------
#     None.

#     '''
#     #load the original network 
#     OG_edgelist = cg.load(networkpath +'/'+'0_reducedEdgelist')
#     OG = cg.load_network(OG_edgelist[:,0:2])

#     #spreading on original network and weight network 
#     sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
#     [netIndex, netNweight]= cg.NetworkInx(sizeCliques,networkpath)
#     alphas = np.arange(0,1.01,0.05)
    
#     results=[]
#     for Sclique in Scliques:
#         print('Sclique:',Sclique)
#         edgelist = cg.load(networkpath +'/' + str(netIndex[Sclique]) +'_reducedEdgelist')
#         RG = cg.weightNetwork(edgelist)
#         arg=[Sclique,OG,RG,alphas,simulations,networkpath]
#         result = cg.ProbOneReducedNet(arg)
#         pk.dump(result,open(resultpath +'/'+str(Sclique)+'_vectorD','wb'))     
    
#         results.append(result)
#     pk.dump(results,open(resultpath +'/VS','wb'))

def SpreadProbValue(networkpath,resultpath,Scliques,simulations):
    '''
    calculate the probability of being infected for each node in k-clique CGNs

    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    Scliques : int
        value of (k-1)-clique.
    simulations : int
        simulation times.

    Returns
    -------
    None.

    '''
    #load the original network 
    OG_edgelist = cg.load(networkpath +'/'+'0_reducedEdgelist')
    OG = cg.load_network(OG_edgelist[:,0:2])
    
    #spreading on original network and weight network 
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex, netNweight]= cg.NetworkInx(sizeCliques,networkpath)
    alphas = np.arange(0,1.01,0.05)
    
    results=[]
    for Sclique in Scliques:
        print('Sclique:',Sclique)
        edgelist = cg.load(networkpath +'/' + str(netIndex[Sclique]) +'_reducedEdgelist')
        RG = cg.weightNetwork(edgelist)
        arg=[Sclique,OG,RG,alphas,simulations,networkpath]
        result = cg.ProbValue_OneReducedNet(arg)
        pk.dump(result,open(resultpath +'/Inf_Prob/'+str(Sclique)+'_ProbValue','wb'))     
    
        results.append(result)
    pk.dump(results,open(resultpath +'/Inf_Prob/All_ProbValue','wb'))   
    
def spreadProbBeta(networkpath,resultpath,beta,simulations):
    '''
    for a given infection rate
    calculate the distance of vector consisting of being infected probability of each node in k-clique  
    
    Parameters
    ----------
    networkpath : str
        path to load the network.
    resultpath : str
        path to save the result.
    beta : list
        infection rate.
    simulations : float
        simulation times.

    Returns
    -------
    None.

    '''
    #spreading on original network and weight network 
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex, netNweight]= cg.NetworkInx(sizeCliques,networkpath)
    
    #load the original network 
    OG_edgelist = cg.load(networkpath +'/'+'0_reducedEdgelist')
    OG = cg.load_network(OG_edgelist[:,0:2])
    
    results=[]
    for clique in sorted(netIndex.keys()):
        
        index = netIndex[clique]
        print('The clique:',clique,'index:',index)
        edgelist = cg.load(networkpath +'/' + str(index) +'_reducedEdgelist')
        RG = cg.weightNetwork(edgelist)
        arg=[clique,OG,RG,beta,simulations,networkpath]
        result = cg.ProbOneReducedNet(arg)
        pk.dump(result,open(resultpath +'/SpreadProbBeta/'+str(clique)+'_BetaT_vectorD','wb'))        
        
        #load the network 
        results.append(result)
    pk.dump(results,open(resultpath +'/SpreadProbBeta/BetaT_vectorD','wb'))

def PlotAxes(ax,xlabel,ylabel,title='',fontsize=20, n_legend = 18, mode=True):
    
    font_label = {'family': "Calibri", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)
    if mode == True:
        ax.legend(loc='best',framealpha=0, fontsize=n_legend)
        
def PlotWeightSpread(ax,path,Scliques):
    '''
    Plot the accuracy of spreading on k-clique CGNs

    Parameters
    ----------
    ax : axes
        axes.
    path : str
        path to load the results.
    Scliques : int
        value of k-clique.

    Returns
    -------
    None.

    '''
    colors = plt.get_cmap('Paired')    
    mec_color = 'black'#'#978C86'#'white'
    n_legend = 18
    markersize = 10
    alp = 1
    mew = 0.8
    
    #load the dataset
    Omcs = cg.load(path +'/simulation/Omcs')
    #Rmcs = cg.load(path + '/Rmcs')
    Rwmcs = cg.load(path+'/simulation/Rwmcs')
    
    omcs = Omcs/Omcs[-1,0]
    #rmcs = Rmcs/Rmcs[-1,0]
    rwmcs = Rwmcs/Rwmcs[-1,0]
    omc = np.average(omcs,axis =1)

    alphas = np.arange(0,1.01,0.01)
    x = np.arange(0,101,5)
    betac = np.array([0.83,0.76,0.71,0.66,0.61,0.58])
    betac_inx = np.array(list(map(lambda x:int(x*100), betac)))

    for i,sclique in enumerate(Scliques):
        ax.plot(alphas[x],rwmcs[:,i][x],'o',ms=markersize, color=colors(i),alpha=alp, mec=mec_color,mew=mew,label = 'k='+str(sclique+1))
        
    ax.plot(alphas[x],omc[x],'-',lw=2,color='black',label='O')
    cg.PlotAxes(ax,r'$\beta$',r'$\rho$','a')
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)

    #subfig1
    axinx = ax.inset_axes((0.60,0.18,0.35,0.35))
    for i,sclique in enumerate(Scliques):
        axinx.plot(omc[betac_inx[i]],rwmcs[betac_inx[i],i],'o',color=colors(i),mec=mec_color,alpha=alp,mew=mew,ms=7)
    
    axinx.plot(omc[betac_inx],omc[betac_inx],color='black',alpha=alp)
    PlotAxes(axinx,r'$\rho^O$',r'$\rho^R$','',fontsize=16, n_legend = 10)
    axinx.set_xlim(0.58,0.9)
    axinx.set_ylim(0.58,0.9)
    axinx.tick_params(direction='in', which='both',length =3, width=1, labelsize=14)

def PlotWeightProbVector(ax,path,Scliques):
    '''
    Plot the accuracy of spreading on k-clique CGNs using the vector distance

    Parameters
    ----------
    ax : axes
        axes.
    path : str
        path to load the result.
    Scliques : float
        value of k-clique.

    Returns
    -------
    None.

    '''
    colors = plt.get_cmap('Paired')    
    mec_color = 'black'#978C86'#'white'
    markersize = 10
    alp = 1
    mew = 0.8
    n_legend = 18
    
    #beta_l = np.array([0.83,0.76,0.71,0.66,0.61,0.58])
    #mec=mec_color
    alphas = np.arange(0,1.01,0.05)
    #kt = cg.load(path+'/VS')
    #LoadProbValue(path,Scliques,alphas)
    
    Avg_Dprob_weight = cg.load(path+'/Inf_Prob/Avg_Dprob_weight')
    Std_Dprob_weight = cg.load(path+'/Inf_Prob/Std_Dprob_weight') 
    
    Avg_Pearson_weight = cg.load(path+'/Inf_Prob/Avg_Pearson_weight')
    Std_Pearson_weight = cg.load(path+'/Inf_Prob/Std_Pearson_weight') 
    
    #subfig1
    axinx = ax.inset_axes((0.55,0.58,0.35,0.35))
    
    for i,sclique in enumerate(Scliques):
        #ax.errorbar(alphas,kt[i][2],yerr = kt[i][3],fmt ='o-',capthick=2,capsize=5,ms= markersize,mec=mec_color,color=colors(i),alpha=alp,mew=mew)
        #ax.vlines(beta_l[i],ymin=0.0, ymax=1, color = colors(i), linewidth=1.5,linestyle='--')
        ax.errorbar(alphas,Avg_Dprob_weight[sclique],yerr = Std_Dprob_weight[sclique],fmt ='o-',capthick=2,capsize=5,ms= markersize,mec=mec_color,color=colors(i),alpha=alp,mew=mew)
        axinx.errorbar(alphas[0:-1],Avg_Pearson_weight[sclique][0:-1],yerr = Std_Pearson_weight[sclique][0:-1],fmt ='o-',capthick=2,capsize=5,ms=7,mec=mec_color,color=colors(i),alpha=alp,mew=mew)
    
    axinx.set_xlim(-0.05,1.05)
    axinx.set_ylim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)

    PlotAxes(axinx,r'$\beta$',r'$r$','',fontsize=16, n_legend = 10)
    cg.PlotAxes(ax,r'$\beta$',r'$D_{Prob}$','b')#r'|$P_R^{inf}-P_O^{inf}|$'
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)

def PlotWeightPc(ax,path,Scliques):
    '''
    Plot the critical point on k-clique CGNs

    Parameters
    ----------
    ax : axes
        axes.
    path : str
        path to load the result.
    Scliques : int
        value of k-clique.

    Returns
    -------
    None.

    '''
    RWpc = {}
    RWcv = {}
    alphas = np.arange(0,1.01,0.01)
    spread = np.zeros((0,101))
    colors = plt.get_cmap('Paired')    
    
    axinx = ax.inset_axes((0.30,0.58,0.35,0.35))
    for i,Sclique in enumerate(Scliques):
        
        OSpread = cg.load(path+'/simulation/'+str(Sclique)+'_Onetmc')
        spread = np.row_stack((spread,OSpread))
        RWSpread = cg.load(path+'/simulation/'+str(Sclique)+'_RWGnetmc')
        [RWcv[Sclique],RWpc[Sclique]] = cg.IdentifyCriInfectRate(RWSpread.T)
        #ax.plot(alphas[1:],RWcv[Sclique][1:],color=colors(i),linewidth=1.5)
        ax.axvspan(RWpc[Sclique],RWpc[Sclique],color=colors(i),linewidth=1.2,linestyle='solid',label =r'$\beta_c^{k'+str(Sclique+1)+'}$')
        axinx.axvspan(RWpc[Sclique],RWpc[Sclique],color=colors(i),linewidth=1.2,linestyle='solid')
        print(Sclique,RWpc[Sclique])
 
    axinx.set_xscale('log')
    [cv,pc] = cg.IdentifyCriInfectRate(spread.T)
    ax.plot(alphas[1:],cv[1:],color='black',linewidth=1.2,linestyle='solid',label = r'$\beta_c^{O}$')
    axinx.plot(alphas[1:],cv[1:],color='black',linewidth=1.2,linestyle='solid')
    ax.axvspan(pc,pc,color='black',linewidth=1.2)
    axinx.axvspan(pc,pc,color='black',linewidth=1.2)

    #fontsize=20 
    n_legend = 16
    #font_label = {'family': "Calibri", 'size':fontsize}
    #ax.set_xlabel(r'$\beta$',  fontdict = font_label)
    #ax.set_ylabel(r'$\chi$', fontdict = font_label)
    #ax.set_title('c', loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    #ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=n_legend)
    ax.legend(loc='upper right',framealpha=0, fontsize=n_legend)
    PlotAxes(axinx,r'$\beta$',r'$\chi$',fontsize=16,n_legend=10)
    cg.PlotAxes(ax,r'$\beta$',r'$\chi$','c')

def BetaTtoNetSize(theorypath,resultpath,networkpath):
    '''
    Calculate the ratio of reduced nodes and edges in the k-clique CGNs

    Parameters
    ----------
    theorypath : str
        path to load the result of spreading on k-clique graph.
    resultpath : str
        path to save the ratio of reduced nodes and edges.
    networkpath : str
        path to load the network.

    Returns
    -------
    Nnode : array 
        the number of nodes in k-clique CGNs.
    Nedge : array
        the number of edges in k-clique CGNs.
    netIndex : dict
        map relation between k-clique CGNs and index.

    '''
    simulation_result_10 = cg.load(theorypath+'/400_1_occupy_probability')
    ct10 = cg.InfectCriticalCondition(simulation_result_10)

    Nnode = np.loadtxt(networkpath+'/Nnodes.txt')
    Nedge = np.loadtxt(networkpath+'/Nedges.txt')
        
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex,Nweight] = cg.NetworkInx(sizeCliques,networkpath)
    betaTtoNnode = {}
    betaTtoNedge = {}
    for kclique in sorted(ct10.keys()):
        betaT = ct10[kclique]
        if kclique-1 in set(netIndex.keys()):
            index = netIndex[kclique-1]
            betaTtoNnode[betaT] = Nnode[index]/Nnode[0]
            betaTtoNedge[betaT] = Nedge[index]/Nedge[0]
        elif kclique-1 == 1:
            betaTtoNnode[betaT]= 1/Nnode[0]
            betaTtoNedge[betaT]= 1/Nedge[1] 
            
    pk.dump(betaTtoNnode,open(resultpath + '/betaTtoNnode','wb'))
    pk.dump(betaTtoNedge,open(resultpath + '/betaTtoNedge','wb'))
    
    return Nnode,Nedge,netIndex

def PlotBtNnode(ax,resultpath,index,name):
    '''
    Plot the ratio of reduced node in the k-clique CGNs

    Parameters
    ----------
    ax : axes
        axes.
    resultpath : str
        path to load the result.
    index : int
        index of color.
    name : str
        label.

    Returns
    -------
    None.

    '''
    n_legend = 18
    colors = plt.get_cmap('Paired')    
    
    betaTNnode = cg.load(resultpath+'/betaTtoNnode')
    ax.plot(betaTNnode.keys(),betaTNnode.values(),color = colors(2*index+1),label=name,ls='--')
    cg.PlotAxes(ax,r'$\beta$',r'$\frac{N_r}{N_o}$(node)','d')
    ax.legend(loc='lower left',framealpha=0, fontsize=n_legend)

def PlotBtNedge(ax,resultpath,index,name): 
    '''
    Plot the ratio of reduced edges in the k-clique CGNs

    Parameters
    ----------
    ax : axes
        axes.
    resultpath : str
        path to load the result.
    index : int
        index of color.
    name : str
        label.

    Returns
    -------
    None.

    '''
    #n_legend = 18
    colors = plt.get_cmap('Paired')    

    betaTNedge = cg.load(resultpath+'/betaTtoNedge')
    ax.plot(betaTNedge.keys(),betaTNedge.values(),color = colors(2*index+1),label=name)
    cg.PlotAxes(ax,r'$\beta$',r'$\frac{E_r}{E_o}$(edge)','e')
    #ax.legend(loc='upper right',framealpha=0, fontsize=n_legend)

def PlotDpRsize(ax,resultpath,Scliques,netIndex,Nnode,Nedge,i):
    '''
    Plot the accuracy for different ratio of reduced nodes and edges in the k-clique CGNs

    Parameters
    ----------
    ax : axes
        axes.
    resultpath : str
        path to load the results.
    Scliques : int
        value of k-clique.
    netIndex : dict
        map realtion between network and index.
    Nnode : array
        the number of nodes in the k-clique CGNs.
    Nedge : array
        the number of edges in the k-clique CGNs.
    i : int
        index of color.

    Returns
    -------
    None.

    '''
    markersize = 10
    mew = 0.8
    beta01_node = {}
    beta01_edge = {}
    point_node = {}
    point_edge = {}
    if i == 0:
        ax.text(0.07,0.9,r'$\beta=0.58$',color='#949396',size=20)

    colors = plt.get_cmap('Paired')    
    for scq in netIndex.keys():
        
        #for different size of k-clique CGNs
        vs = cg.load(resultpath+'/SpreadProbBeta58/'+str(float(scq))+'_BetaT_vectorD')[2]
        index = netIndex[scq]
        
        beta01_node[vs[0]] = Nnode[index]/Nnode[0]
        beta01_edge[vs[0]] = Nedge[index]/Nedge[0]
        if int(scq) == 9:
           point_node[vs[0]] =  Nnode[index]/Nnode[0]
           point_edge[vs[0]] =  Nedge[index]/Nedge[0]
   
    #xe = list(point_edge.keys())[0]
    #ye = list(point_edge.values())[0]
    #ax.axvline(xe,color=colors(2*i+1),linestyle='dashed',lw=1)#'#949396'
    
    #ax.plot(beta01_node.keys(),beta01_node.values(),'o-',color=colors(2*i+1))
    ax.plot(beta01_node.keys(),beta01_node.values(),color=colors(2*i+1),linestyle='dashed')
    ax.plot(point_node.keys(),point_node.values(),marker='s',mec = 'black',color=colors(2*i+1), ms = markersize,mew=mew)
    #xn = list(point_node.keys())[0]
    #ax.axvline(xn,color='gray',linestyle='dashed')
    #ax.text(xn+0.002,0.2,r'$D_{prob}^{\beta^t=\beta}$',color='gray',size=15)
    
    ax.plot(beta01_edge.keys(),beta01_edge.values(),color=colors(2*i+1),linestyle='solid')
    ax.plot(point_edge.keys(),point_edge.values(),marker='o',mec ='black',color=colors(2*i+1), ms = markersize,mew=mew)

    ax.set_xscale("log")
    ax.set_xlim([0.01,0.18])       

    cg.PlotAxes(ax,r'$D_{prob}$',r'$\frac{E_r}{E_o}$,  $\frac{N_r}{N_o}$','f')    
    #PlotAxes(axes2,'',r'$\frac{E_r}{E_o}$(edge)')
            
def PlotVerify(resultpath,theorypath,networkpath,figurepath,files):
    '''
    Plot the accuracy of spreading on k-clique CGNs

    Parameters
    ----------
    resultpath : str
        path to load the results.
    theorypath: str
        path to load the results on k-clique structure
    networkpath: str
        path to load the k-clique network
    figurepath: str
        path to save the figure
    files : list
        dataset names.

    Returns
    -------
    None.

    '''
    Scliques = [4,5,6,7,8,9]
    fig,ax = plt.subplots(2,3,figsize=(18,10),constrained_layout=True)

    path = resultpath+'/'+files[0] #path to load the result of GrQc 
    PlotWeightSpread(ax[0,0],path,Scliques)
    PlotWeightProbVector(ax[0,1],path,Scliques)
    PlotWeightPc(ax[0,2],path,Scliques)
    
    #ax[1,1].axvline(0.58,color='#949396',linestyle='dashed',lw=1)
    #ax[1,0].axvline(0.58,color='#949396',linestyle='dashed',lw=1)

    for i,each in enumerate(files):
        
        #for each dataset
        print('file:',each)
        eachrespath = resultpath+ '/' + each
        eachnetpath  = networkpath + '/' +each
        [Nnode,Nedge,netIndex]= BetaTtoNetSize(theorypath,eachrespath,eachnetpath)
    
        PlotBtNnode(ax[1,0],eachrespath,i,each)
        PlotBtNedge(ax[1,1],eachrespath,i,each)
        
        if i == 0:
           #axes2 =ax[1,2].twinx()
           PlotDpRsize(ax[1,2],eachrespath,Scliques,netIndex,Nnode,Nedge,i)
           ax[1,0].sharey(ax[1,2])
           ax[1,1].sharey(ax[1,2])
        else:
           PlotDpRsize(ax[1,2],eachrespath,Scliques,netIndex,Nnode,Nedge,i)
    
    plt.savefig(figurepath+'/figure4.png',dpi=600)
    plt.savefig(figurepath+'/figure4.eps')
    plt.savefig(figurepath+'/figure4.pdf')
    
if __name__ == '__main__':
    
  networkpath = root_path + '/NetworkReduction/fig3_reductionSize/network'  
  resultpath = root_path + '/NetworkReduction/fig4_reductionAccuracy/result'
  figurepath = root_path + '/NetworkReduction/fig4_reductionAccuracy/figure'
  theorypath = root_path + '/NetworkReduction/fig2_leastInfProb/result'

  #Four part- Spreading
  #set the parameter
  files = ['GrQC','CondMat','HepPh','NetScience']
  Scliques = [4,5,6,7,8,9]
  simulations = 1000
  
  #begin to spread. 
  #please note that there are all results about spreaidng in existing file  
  #do not excute the 'for' block if only intend to plot the result, it will takes a lot of times to run the code and cover existing spreading result
  #therefore, please ignore the sentence consiting of 'for' and directedly excutre the PlotVerify
  
  # for file in files:
      
  #     #run the coarse-graining for each datasets
   
  #     #1.MC spreading
  #     eachNetpath = networkpath + '/' + file
  #     simuResultpath = resultpath + '/' + file + '/simulation'
  #     SpreadSimulations(eachNetpath,simuResultpath,Scliques,simulations)  
   
  #     #2.InfProbValue
  #     ProbResultpath = resultpath + '/' + file
  #     #SpreadProb(eachNetpath,ProbResultpath,Scliques,simulations)
  #     SpreadProbValue(eachNetpath,ProbResultpath,Scliques,simulations)
 
  #     #3.ProbBeta
  #     betas = [0.36,0.83]#0.58
  #     spreadProbBeta(eachNetpath,ProbResultpath,betas,simulations)

  #4.Plot MC spreading and KL-divergence
  PlotVerify(resultpath,theorypath,networkpath,figurepath,files)


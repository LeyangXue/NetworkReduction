# NetworkReduction

Codes as part of paper "Iterative structural coarse graining for propagation dynamic in complex network"

The code is used to perform the numerical simulation and visualization. For more detailed information, please see the paper. 

## Describtion

The project propose a new iterative structural coarse-graining method to reduce the size of network and preserve the dynamical behavior of SIR model.

The iterative structural coarse-graining method is applied to the problem of realistic propagation, such as influence maximization, edge immune and surveilance sentinel, which provides a new perspective to identify the important structural information. 

## Content 

All scripts used to perform the numerical simulation and plot the figure of the manuscript are provided in each directory. Contents are divided into different sub directory, such as network, code, result, and figure.

* code  
The file contains the code (main.py) to run the numerical simulation and plot the figure, please correct the current path if run the code

* network  
The file contains the original network, coarse-grained network, and information about coarse-graining process 

* result  
The file contains the result of numerical simulation

* figure  
The file contains the figure 

All common codes are organized into the packages, e.g. coarse_grain.py, lised in utils files.
The utils contains four packages:

* coarse_grain.py  
The package includes all functions to run the network coarse-graining process and SIR numerical simulation, load and save the datasets, and calculate the probability of node being infected in the k-clique CGNs, and so on.

* kplexes.py  
The package is used to identify the k-plex structure in the network. 
The package can be ignored if do not run the approximately coarse-graining code

* prunconnected.py  
The package provides the basic function for kplexes.py.  

* noprun.py  
The package provides the basic function for kplexes.py.  

## Install and Run

You can install or download the NetworkReduction to the local.

* Clone the repository  
$ git clone https://github.com/LeyangXue/InnovationDiffusion.git

* Dependence 
  + snap 
  + copy 
  + sets
  + itertools
  + tool
  + pickle
  + networkx 
  + math
  + collections
  + uuid 
  + random 
  + multiprocessing 
  + datetime 
  + sys 
  + numpy 
  + pandas 
  + scipy
  + powerlaw 
  + sympy   

  please check the dependence of package before runing the code. If you need to install the packages: pip install (package)  

* change the value of root_path  in each script as current local path  
root_path  = '/current_path'

## Email

Any suggestion are welcome and please send your suggestion to hsuehleyang@gmail.com



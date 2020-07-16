# Prioritized Restreaming Algorithms for Balanced Graph Partitioning
Source code for [the paper](https://arxiv.org/pdf/2007.03131.pdf) by Amel Awadelkarim and Johan Ugander. 

For questions, please email Amel at ameloa@stanford.edu.

In this repository, we present our python implementations of three methods for balanced graph partitioning --- Balanced Label Propagation (Ugander and Backstrom, 2013), Social Hash partitioner (Kabiljo et al, 2017; Shalita et al, 2018), and Restreamed Linear Deterministic Greedy (Nishimura and Ugander, 2013). ReLDG is accelerated using Cython; to build Cython code, execute `python setup.py build_ext --inplace` from within the `/src` directory. Otherwise, these implementations are not optimized for performance but for analyzing various design decisions of each.

We used the following versions of external python libraries:
* `cvxpy==1.1.1`
* `Cython==0.29.20`
* `matplotlib==3.1.3`
* `numpy==1.18.1`
* `scipy==1.4.1`
* `seaborn==0.9.0`

To generate the results for one of the networks in the paper:
1. Clone repository.
2. Download and save your test network as a .txt file to the `/data` folder. In the paper we report results on the following datasets, all from the Stanford Network Analysis Project (Leskovec and Krevl, 2014):

    * [soc-Pokec](http://snap.stanford.edu/data/soc-Pokec.html) 
    * [com-LiveJournal](http://snap.stanford.edu/data/com-LiveJournal.html) 
    * [com-Orkut](http://snap.stanford.edu/data/com-Orkut.html) 
    * [web-NotreDame](http://snap.stanford.edu/data/web-NotreDame.html) 
    * [web-Stanford](http://snap.stanford.edu/data/web-Stanford.html) 
    * [web-Google](http://snap.stanford.edu/data/web-Google.html) 
    * [web-BerkStan](http://snap.stanford.edu/data/web-BerkStan.html)

3. Navigate to the Jupyter notebook at `/src/paper_figures.ipynb` and execute all cells using a Python 3 kernel, adapting cell 4 to your network of choice. Note that if a network outside of our list is selected, its number of edges must be recorded in cell 4 of the notebook.

To generate the incumbency plot in the paper (Figure 2: Final partition quality vs Gain threshold), we must run each method with the incumbency parameter sweeping [-10, 10]. To automate this, we include a shell script `incumbency.sh` that is executed in cell 29 of the notebook. Please update the network name here, if other than web-NotreDame.

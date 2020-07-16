# Prioritized Restreaming Algorithms for Balanced Graph Partitioning
Source code for paper by Amel Awadelkarim and Johan Ugander. 

For questions, please email Amel at ameloa@stanford.edu.

In this repository, we present our implementations of three methods for balanced graph partitioning --- Balanced Label Propagation (Ugander and Backstrom, 2013), Social Hash partitioner (Kabiljo et al, 2017; Shalita et al, 2018), and Restreamed Linear Deterministic Greedy (Nishimura and Ugander, 2013). These methods are not optimized for performance but for analyzing various design decisions of each.

To generate the plots in the paper:

1. Clone repository.
2. Download and save your desired network as a .txt file, formatted as those from the Stanford Network Analysis Platform (SNAP), to the `/data` folder. In the paper we report on results for the following datasets:
  - [soc-Pokec](http://snap.stanford.edu/data/soc-Pokec.html), 
  - [com-LiveJournal](http://snap.stanford.edu/data/com-LiveJournal.html) 
  - [com-Orkut](http://snap.stanford.edu/data/com-Orkut.html), 
  - [web-NotreDame](http://snap.stanford.edu/data/web-NotreDame.html), 
  - [web-Stanford](http://snap.stanford.edu/data/web-Stanford.html), 
  - [web-Google](http://snap.stanford.edu/data/web-Google.html), and 
  - [web-BerkStan](http://snap.stanford.edu/data/web-BerkStan.html),
3. Navigate to the Jupyter notebook at `/src/paper_figures.ipynb` and execute all cells with a Python 3 kernel.



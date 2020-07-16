# Prioritized Restreaming Algorithms for Balanced Graph Partitioning
Source code for paper by Amel Awadelkarim and Johan Ugander. For questions, please email Amel at ameloa@stanford.edu.

In this repository, we present our implementations of three methods for balanced graph partitioning, all in Python 3 --- Balanced Label Propagation (Ugander and Backstrom, 2013), Social Hash partitioner (Kabiljo et al, 2017; Shalita et al, 2018), and Restreamed Linear Deterministic Greedy (Nishimura and Ugander, 2013). 

To generate the plots in the paper:

1. Clone repository.
2. Download and save your desired network as a .txt file, formatted as those from the Stanford Network Analysis Platform (SNAP), to the `/data` folder. In the paper we report on results for soc-Pokec, com-LiveJournal, com-Orkut, web-NotreDame, web-Stanford, web-Google, and web-BerkStan.
3. Navigate to the Jupyter notebook at `/src/paper_figures.ipynb` in and execute all cells with a Python 3 kernel.

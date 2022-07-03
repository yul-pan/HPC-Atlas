# HPC-Atlas

HPC-Atlas: Computationally Constructing A Comprehensive Atlas of Human Protein Complexes

Please install the python dependency package before running the clustering program, such as networkx, numpy and so on.

**input**:
      tab separated protein interaction network: portein-1 [tab] proein-2 [tab] weight

**output**:
      each row represents a protein complex: protein-1 [tab] ,..., protein-N

**Example**
```
pyhton3 cluster.py --input_PPI_weight ../Data/test_data/test_weight.txt --outfile ../test_outfile.txt
```

The "--input_PPI_weight" is the input weighted PPI network file.

The "--outfile" is the final protein complexes file.

The outfile_first is the cache file of intermediate results, which can be deleted after the program is completed.

Because the PPI file of the whole network is too large, it can only be put into Releases module and can be downloaded here: https://github.com/yul-pan/HPC-Atlas/releases

The feature matrix used in the experiment can be downloaded: https://share.weiyun.com/BFZckLjR

Since GitHub restricts the upload file to no more than 25M, please download the full version (code, data and results file) here:https://share.weiyun.com/oXupED6J

For convenience to relevant researchers, all experimental data and result files are presented in the form of UniProtID and Gene name, respectively.

# Federated Learning on Knowledge Graphs

PyTorch code that accompanies the paper [Efficient Federated Learning on Knowledge Graphs via Privacy-preserving Relation Embedding Aggregation](https://arxiv.org/pdf/2203.09553.pdf).
This repo is built upon previous work **FedE** ([paper](https://arxiv.org/pdf/2010.12882.pdf) and [code](https://github.com/AnselCmy/FedE)). 
Our work contains Knowledge Graph Reconstruction Attack and federated version of the following Knowledge Graph algorithms: TransE, RotatE, ComplEx, DistMult, NoGE and KB-GAT.

## Prepare Dataset: 
There is a data preprocessing file "dataset.ipynb" which allows split triples randomly. 
We does not provide original data here which can be downloaed online, but we provide a federated version of DDB14 in folder `--Fed_data`.

## Run Experiments: 
There is script "exp.sh" which allows running all federated experiments with non-gnn models, while the codes of Fed-NoGE are in folder `--NoGE`.
One code demo of reconstruction attack is shown in `rec_attack_fedr_fb15k_TransE.ipynb`.

### Citation
Please cite our paper if you find this code useful for your research.
For any clarification, comments, or suggestions please create an issue or contact deepakn1019@gmail.com

# Federated Learning on Knowledge Graphs

PyTorch code that accompanies the paper [Efficient Federated Learning on Knowledge Graphs via Privacy-preserving Relation Embedding Aggregation](https://arxiv.org/pdf/2203.09553.pdf). Our work contains **Knowledge Graph Reconstruction Attack** and the federated version of the following Knowledge Graph algorithms: TransE, RotatE, ComplEx, DistMult, NoGE and KB-GAT.

### Prepare Dataset: 
There is a data preprocessing file `dataset.ipynb` which allows split triples randomly. 
We does not provide original data that can be downloaed online, but we provide a federated version of DDB14 in folder `--Fed_data`.

### Run Experiments: 
There is script `exp.sh` running all federated experiments with non-gnn models, while the codes of Fed-NoGE are in folder `--NoGE`.
Some examples of reconstruction attack are shown in `rec_attack_fedr_fb15k_TransE.ipynb` as well as in the folder `--FedE/Rec_Attack/..`.

### Citation
Please cite our paper if you find this code useful for your research.
For any clarification, comments, or suggestions please create an issue or contact kaz321@lehigh.edu

```
@article{fedr,
      title={Efficient Federated Learning on Knowledge Graphs via Privacy-preserving Relation Embedding Aggregation}, 
      author={Zhang, Kai and Wang, Yu and Wang, Hongyi and Huang, Lifu and Yang, Carl and Chen, Xun and Sun, Lichao},
      journal={Findings of EMNLP},
      year={2022},
}
```

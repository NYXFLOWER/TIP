# FM-PSEP
Our hierarchical graph convolutional neural network model *FM-PSEP* is a 
general approach for multirelational link prediction in any hierarchical 
heterogeneous network. 

Here, Here, we are particularly concerned about the 
safety of [polypharmacy](https://en.wikipedia.org/wiki/Polypharmacy), which is 
the concurrent use of multiple medications by a patient. Given a pair of drug, 
the model will predict how many polypharmacy side effects the drug pair will 
have, and what are the possibilities.

![](img/pred_dd.png)

Code implementations base on the [PyTorch-Geometric](https://github
.com/rusty1s/pytorch_geometric) package, make sure it has been installed 
before running the code.

## Data Drive

We construct a hierarchical heterogeneous graph of two node types: proteins and 
drugs. We think of the whole graph as three sub-graph.
- *p-net*: protein-protein association network
- *d-net*: drug-drug association network with multiple edge types (each edge 
type refers to a polypharmacy side effect)
- *pd-net*: drug-protein association network whose edges have the direction 
from protein node to drug node
![](img/network.png)
![](img/model.png)


## Performance Comparision

We will firstly evaluate three auto-encoder approach (Hamilton et al., 2017), and compare them with our approach:
- GAE (Kipf and Welling, 2016): [[paper]](https://arxiv.org/abs/1611.07308)
- AROPE (Zhang et al., 2018): [[paper]](https://dl.acm.org/citation.cfm?id=3219969), [[code]](https://github.com/ZW-ZHANG/AROPE)
- LINE (Tang et al., 2015): [[paper]](https://arxiv.org/abs/1503.03578)

We firstly consider using DistMult (Yang et al., 2015[[paper]](https://arxiv.org/abs/1412.6575)) as decoder to predict the polypharmacy side effect of drug pairs.

Then, we will compare our model with the state-of-the-art polypharmacy side effect prediction models:
- DECAGON (Zitnik, 2018): [[paper]](https://arxiv.org/abs/1802.00543), [[code]](https://github.com/marinkaz/decagon)
- DeepDDI (Ryu et al., 2018): [[paper]](https://www.pnas.org/content/115/18/E4304), [[code]](https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/)
- mvGAE (Ma et al., 2018): [[paper]](https://arxiv.org/abs/1804.10850), [[code]](https://github.com/matenure/mvGAE)


![](img/step.png)

## Result by now
Pre-train ppi embedding with 2 layers GCN for 80 epochs([output](./out/ppp.pdf)): 
- around 92% auprc
- less than 2GB GPU memory cost. 
- Time cost: 2 mins.


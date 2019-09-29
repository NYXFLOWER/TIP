# Tri-graph Information Propagation (TIP) model

TIP is an efficient general approach for **multi-relational link prediction** in any **multi-modal**  (i.e. heterogeneous and multi-relational) network with two types of nodes. It can also be applied to the **Knowledge Graph Completion** and **Recommendation** task. TIP model is inspired by the [Decagon](https://github.com/marinkaz/decagon) and [R-GCN](https://github.com/tkipf/relational-gcn) models, motivated by their limitations of high computational cost and memory demand when graph goes really complex. TIP improves their link prediction **accuracy**, and time and space **efficiency** of node representation learning. See details on the algorithm in our [paper]().

## TIP for Polypharmacy Side Effiect Prediction

we are particularly concerned about the safety of [polypharmacy](https://en.wikipedia.org/wiki/Polypharmacy), which is the concurrent use of multiple medications by a patient. Given a pair of drug (:pill:,:pill:), the TIP model will predict how many polypharmacy side effects the drug pair will have, and what are the possibilities.

<div align=center>
<img height="100" src="img/pred_dd.png" alt=""hhh/>
</div>

We use *POSE clinical records* and *pharmacological information* to construct a multi-modal biomedical graph with two types of nodes: Drug (D) and Protein (P). The graph contains three types of interaction (refer to three subgraphs): 

&emsp; :cookie: &ensp; D-D graph: drug-drug interactions with side effects as edge labels

&emsp; :cake: &ensp; P-D graph: protein-drug interactions (with a fixed label)

&emsp; :ice_cream: &ensp; P-P graph: protein-protein interactions (with a fixed label)

<div align=center>
<img width="500" src="img/network.png" alt=""hhh/>
</div>

TIP model embeds proteins and drugs into different spaces of possibly different dimensions in the encoder, and predict side effects of drug combinations in the decoder. As shown below, TIP learns the protein embedding firstly on the P-P graph, and passes it to D-D graph via D-P graph. On D-D graph, TIP learns drug embedding and predicts relationships between drugs.

**TIP Encoder**:

<div align=center>
<img height="300" src="img/encoder.png">
</div>

**TIP Decoder**:

<div align=center>
<img height="300" src="img/decoder.png">
</div>

## Source Code

TIP is implemented in PyTorch with [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) package. It is developed and tested under Python 3.  

### Requirement

TIP is trained and tested on **GPU**. Make sure the essential packages in `requirements_gpu.txt` has been installed in your environment properly. Use the following command to install all the required packages:

```shell
$ pip install -r requirements_gpu.txt
```

If you do want to run the code with **CPU**, install the packages in the `requirements_cpu.txt`. Then, comment or remove `@profile` before `train()` function in the python file you want to run.

```python
##################################################
@profile        # remove this for training on CPU
##################################################
def train():
```

### Running

The processed data and the code for data processing are in the `./data/` folder. The raw datasets are available on the [BioSNAP](http://snap.stanford.edu/biodata/index.html). See `./data.ipynb` for the full polypharmacy datasets analysis.

The `./model/` folder contains two TIP implementation examples and four TIP variants. Details on experimental setup can be found in our [paper](). You can run any of them as following:

```shell
$ python [model_name].py
```

By default, it uses a subdataset with only five side effects. Comment or remove the following code to train and test with the full datasets. It appears at the head part of the file.

```python
#########################################################################
et_list = et_list[:3]       # remove this line for full dataset learning
#########################################################################
```

:new_moon_with_face::waxing_crescent_moon::first_quarter_moon::waxing_gibbous_moon: **Please browse/open issues should you have any questions or ideas**​ :waning_gibbous_moon::last_quarter_moon::waning_crescent_moon::new_moon_with_face:

## License

TIP is licensed under the MIT License.
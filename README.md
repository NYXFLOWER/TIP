# FM-PSEP
Our model is implemented using [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) package.


We will firstly evaluate three auto-encoder approach (Hamilton et al., 2017), and compare them with our approach:
- GAE (Kipf and Welling, 2016): [[paper]](https://arxiv.org/abs/1611.07308)
- AROPE (Zhang et al., 2018): [[paper]](https://dl.acm.org/citation.cfm?id=3219969), [[code]](https://github.com/ZW-ZHANG/AROPE)
- LINE (Tang et al., 2015): [[paper]](https://arxiv.org/abs/1503.03578)

![](img/step.png)

Then, we will compare our model with the state-of-the-art polypharmacy side effect prediction models:
- DECAGON (Zitnik, 2018): [[paper]](https://arxiv.org/abs/1802.00543), [[code]](https://github.com/marinkaz/decagon)
- DeepDDI (Ryu et al., 2018): [[paper]](https://www.pnas.org/content/115/18/E4304), [[code]](https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/)
- mvGAE (Ma et al., 2018): [[paper]](https://arxiv.org/abs/1804.10850), [[code]](https://github.com/matenure/mvGAE)




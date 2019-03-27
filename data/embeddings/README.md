# Embeddings

## CGCNN

The following paper describes the details of the CGCNN framework, infomation about how the onehot atom embedding was constructed is availiable in the supplementary materials:

[Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

##MEGnet

The following paper describes the details of the MEGnet framework, the embedding is generated from the atomic weights through one MEGnet layer on a training task to predict the computed formation energies of ∼69,000 materials from the [Materials Project](https://materialsproject.org/) Database:

[Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://arxiv.org/abs/1812.05055)

##MatScholar

This is an experimental NLP embedding based on the data mining of definite compositions and structure prototypes.

[The relevant work is yet to be published, See MatMiner](https://hackingmaterials.github.io/matminer/matminer.utils.html?highlight=matscholar#matminer.utils.data.MatscholarElementData)

##Magpie / Magpy

Magpie is a featurisation technique that takes various functions of atom properties and weights in order to construct a feature vector. Here we give an embedding that uses only the atomic properties from Magpie. 

We have rewritten the Magpie look-up functionality in python and it can be found here: [Magpy](https://github.com/CompRhys/magpy). We have added some additional properties to the data tables not found in the original Magpie tables. This embedding is a simply the elemental lookup values from Magpy. 

[A general-purpose machine learning framework for predicting properties of inorganic materials](https://www.nature.com/articles/npjcompumats201628.pdf)


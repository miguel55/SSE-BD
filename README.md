# SSE-BD: A self-supervised embedding of cell migration features for behavior discovery over cell populations

## Description
This repository contains the code and documentation to implement SSE-BD: A self-supervised embedding of cell migration features for behavior discovery over cell populations, a representation learning technique for the discovery of shared cell behaviors over cell populations from a collection of dynamic features. 

```
A self-supervised embedding of cell migration features for behavior discovery over cell populations,
Miguel Molina-Moreno, Iván González-Díaz, Ralf Mikut, Fernando Díaz-de-María
Computer Methods and Programs in Biomedicine, doi: XXX, 2024. 
```

This code is partly based on the Pytorch implementation of the BELoss from the paper Training deep retrieval models with noisy datasets: Bag exponential loss [https://github.com/tmcortes/BELoss](https://github.com/tmcortes/BELoss).

## License

SSE-BD code is released under the CC BY-NC 4.0 License (refer to the `LICENSE` file for details).

## Citing SSE-BD

If you find SSE-BD useful in your research, please consider citing:

	@ARTICLE{ssebd,
		title = {A self-supervised embedding of cell migration features for behavior discovery over cell populations},
		journal = {Computer Methods and Programs in Biomedicine},
		volume = {},
		pages = {},
		year = {2024},
		doi = {XXX},
		author = {Miguel Molina-Moreno and Iván González-Díaz and Ralf Mikut and Fernando Díaz-de-María}
	}
  
## Dataset and models

Our Githbb implementation includes our trained models and original feature database.

## Requirements

SSE-BD is implemented to not require any additional modules. The Python code has been tested with Python 3.8.5, Pytorch 1.3.1 and CUDA 10.1.

Note: The clustering and t-SNE results may differ from the ones presented in the paper for newer versions of Python. We provide our results, including the clustering labels (cell behaviors) and the GMM probabilities for our approach in the folder `examples/results/analysis/embedding_data_16`.

## Demo

To test our approach with the provided database and models, follow the steps below. 

1. First, execute the `evaluation.py` script. It compares our approach with a 16-dimensional dynamic embedding with the original approach with sequences of 21 time steps with 21 features.
2. Second, the `embedding_analysis.py` script provides the visualizations from the paper: embedding, behavior transitions and correlation matrices for explainability.

To train our approach in your data, use the `train_SSE_BD.py` script modifying the parameters according to your data structure.

## Installation

To start using SSE-BD, download and unzip this repository.
```
git clone https://github.com/miguel55/SSE-BD
```

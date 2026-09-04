# Soft Symbol Grounding for Prototypical Concepts

This repository holds the code for our paper *Soft Symbol Grounding for Prototypical Concepts* (Galván-López, Upreti, Aguilar-Ibáñez, Calvo, and Belle).

We introduce **Soft-PNet**, a neuro-symbolic method that grounds latent concepts without a hand-crafted differentiable loss. From a single labeled anchor per concept it builds a prototype distribution in the embedding space, scores a precomputed cache of feasible symbolic solutions against that distribution, refines the cache with a projection-based Metropolis walk, and trains the network against one KL divergence between the prototype-weighted cache and the predicted concepts. The same objective is reused unchanged across tasks.

Alongside Soft-PNet, the repo includes the baselines we compare against: the softened symbol grounding of Li et al. (SoftG) and its K-sample variant (SoftG-K), and a prototypical network trained with Semantic Loss (PNet+SL). Everything runs on three tasks: MNIST-EvenOdd, Visual Sudoku (4x4), and Kand-Logic.

## Installation

The reported results were produced with Python 3.10 and PyTorch 2.2. Install the dependencies into a fresh environment:pip install -r requirements.txt

```
pip install -r requirements.txt
```

## Data

MNIST is downloaded automatically by torchvision the first time you run MNIST-EvenOdd or Visual Sudoku, and the 4x4 Sudoku solution cache is enumerated at runtime, so those two tasks need no extra setup. Kand-Logic expects its preprocessed scene tensors: run `python -m tools.gen_Kandlogic.py` to generate the files (it will take some minutes).

## Layout

* `src/backbones` contain all the backbones used for the models, it is divided by task `Protoencoder` for Prototype training and `lenet` for standard visual recognition.
* `src/databuilder` contain all the pre procesing of the raw datasets into the loaders used for training.
* `src/datasets` contain the raw datasets.
* `src/models` here you can find the individual models organized by task:
* - `softpnet.py`: our propousal.
* - `pnet.py`: is the prototypical network with Semantic Loss (PNet+SL).
* - `softg.py` and `softgk.py`: are the SoftG and SoftG-K baselines.
* `src/samplers/` holds the per-task machinery: the global solution cache, the projection operator and its inverse, the corruption operator, and the Kand-Logic pattern oracle.
* `src/test/` are the experiment runners, and `results/` is where their output lands.

## Citation

If you use this code, please cite the paper:

> Galván-López, Upreti, Aguilar-Ibáñez, Calvo, and Belle. *Soft Symbol Grounding for Prototypical Concepts.* 2026.

A BibTeX entry will be added once the paper is published.

## Download here!

Personal website: [https://magalop-bit.github.io/portafolio/softpnet.html] (It will be replaced when the proceedings are released).

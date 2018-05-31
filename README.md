# Deep-Learning project

This repository contains our work on the first project of the EPFL Deep Learning [course](http://edu.epfl.ch/coursebook/en/deep-learning-EE-559).
The goal was to perform a binary classification on EEG data to predict the laterality of finger movements.

## Workflow

A Neural Network was designed from scratch to perform this task. The used framework was PyTorch.
Moreover the main code implements different baselines using the [scikit-learn](http://scikit-learn.org/stable/) library.

## Indications

The main executable script is the [test.py](test.py) file. By running it, the dataset will be loaded if already installed. If not, it will be downloaded using the [dlc_bci.py](dlc_bci.py) script. The file contains different functions to compute a baseline for the task using an LDA and SVM classifier.
All different informations on the implementation and development of the network is explained in the [report](prediction-finger-movements.pdf).

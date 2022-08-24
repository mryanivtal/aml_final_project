
# Introduction

This repo contains advanced machine learning final project.
The project is based on GP-VAE and contains following parts:
 1. Part 1 - repreducing the original paper
 2. Part 2 - innovation part where we take GP-VAE that orignally designed for imputation for creating synthetic data to enrich our limited training set and evaluate the impact of it by reporting classification model accuracy on the test set.
 
# Getting Started 
## Part 1

Part 1 contains the following parts:
* Project proposal 
* GP_VAE_repreduce notebook that repredocues GP-VAE results
* Model results pdf

## Part 2

Part 2 conatins following parts:
* gp_vae_for_data_generation folder - gp vae source code adjusted to support data generation flow
* GP_VAE_Generation - notebook to train gp-vae with limited train set and generate syntetic data
* timegan_generation - notebook to generate data with TimeGAN
* timeseries_classification - train time-series classification model and evaluate performance

# Code Run
In order to repreduce our results you should clone this repo to your google drive and take the following steps:

## Part 1
Run GP_VAE_repreduce notebook

## Part 2
Run the follwing notebooks in this order:
* GP_VAE_generation - to generated limites train set and generate data with GP-VAE
* Time_GAN_generation - to generate data with TimeGAN
* timeseries_classifictaion - to classify time-series with original limitest train-set, GP-VAE generated data, TimeGAN generated data and see each one impact on test set accuracy


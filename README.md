# Conformal Time Series Forecasting using Reinforcement Learning

This repository contains the implementation for Bhambu et al.'s paper "Conformal Time Series Forecasting using Reinforcement Learning", which has been submitted to NeurIPS 2024.

The codebase extends upon the implementation for "Conformal time-series forecasting" (presented at NeurIPS 2021), originally available at https://github.com/kamilest/conformal-rnn.

## Installation

Python 3.10 or higher is recommended. Install the required dependencies listed in `requirements.txt`.

## Replicating Results

To replicate the experimental results, execute the notebook `example_A2C.ipynb`.

You can obtain the publicly available data used in this work. Note that datasets such as EUR/USD, USD/AUD, USD/GBP, USD/CNY, and USD/CAD require Yahoo Finance credentialing for access.


## Codes

The Python codes are in two formats. The proposed methodologies and compared ones are in the main branch for univariates, while for multi-horizon, the codes are in a multivariable file.


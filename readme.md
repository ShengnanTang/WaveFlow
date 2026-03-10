This code is the official PyTorch implementation of our Paper: WaveFlow: Hierarchical Frequency-Decoupled Flow
Matching for Probabilistic Time Series Forecasting .

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!


## Quickstart

### Installation
```
conda env create -f environment_conda.yml

Notice: It is generally unnecessary to install all the packages listed in the environment_conda.yml file. You can first install Python, followed by core packages such as PyTorch.
```
### Data preparation
Prepare Data. You can obtained the well pre-processed datasets at https://drive.google.com/drive/folders/1l0c4H57xYKKQQ5Tm7kd4C8M2nCepky-y. 
Then place the downloaded data under the folder ```./datasets/```.
```
datasets
|   |-- ETT-small
|   |   |-- ETTh1.csv
```

### Train and evaluate model
config/hyper_parameters_search is used for short-term forecasting.
config/hyper_parameters_search_long is used for long-term forecasting.
### Train and evaluate model
For example you can reproduce a experiment result as the following:
```
cd waveflow
bash config/hyper_parameters_search/electricity/electricity_waveflow_seed_0.sh
```

## Community Support | Acknowledgements
This project is built on the shoulders of the open-source community.  
Special thanks to the **authors and contributors** of the following repository:

- **[K2VAE](https://github.com/decisionintelligence/K2VAE/tree/master)**

## Contact

If you have any questions or suggestions, feel free to contact:
tsn@zju.edu.cn

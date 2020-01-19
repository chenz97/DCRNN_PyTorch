# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This is a PyTorch implementation of Diffusion Convolutional Recurrent Neural Network for the traffic prediction project of Introduction to Data Science, 2019 fall of PKU. This repo. is based on the [DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch) repo.


## Requirements
* torch>=1.2.0
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* tensorflow>=1.3.0
* tables
* future

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Data Preparation
The traffic data files for PeMS-D7, i.e. `train` and `test` folder and `distance.csv`,  should be put in the `data/PEMS-D7` folder.

## Graph Construction
Graph adj matrix is already available in `data/sensor_graph/adj_mx_d7.pkl`. If you want to construct the graph for PeMS-D7 yourself, run the following command:

```bash
cd scripts
python gen_dist_file.py 
python gen_sensor_ids.py
cd ..
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids_d7.txt --normalized_k=0.1\
    --distances_filename=data/sensor_graph/distances_d7.csv
    --output_pkl_filename=data/sensor_graph/adj_mx_d7.pkl
```



## Run the Pre-trained Model on PeMS-D7

(No pre-trained models provided currently.)


## Model Training
```bash
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_d7.yaml
```

There is a chance that the training loss will explode when the learning rate is relatively large (e.g. 0.01), the temporary workaround is to restart from the last saved model before the explosion, or to decrease the learning rate earlier in the learning rate schedule. 

## Model Testing

```bash
python test.py --config_filename=data/model/dcrnn_d7.yaml --epoch=119 --log_dir=adam_mse
```



## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following papers:
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}

@article{yu2017spatio,
  title={Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting},
  author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
  journal={arXiv preprint arXiv:1709.04875},
  year={2017}
}
```

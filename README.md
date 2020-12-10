### Decoupled Appearance and Motion Learning for Efficient Anomaly Detection in Surveillance Video

This repository provides the implementation for our paper [**Decoupled Appearance and Motion Learning for Efficient Anomaly Detection in Surveillance Video** (Bo Li, Sam Leroux, and Pieter Simoens)](https://arxiv.org/abs/2011.05054). We experimentally show that our method achieved higher anomaly detection accuracy and inference speed than the existing works on several benchmark datasets. 

Our anomaly detection framework can be seen in the figure below:
![algorithm](gt/framework.png)



#### Installation and preparation 

1. Clone this repo and prepare the environment:

   ```bash
   git clone https://gitlab.ilabt.imec.be/bobli/daml.git
   cd daml
   ./requirement.sh download_ckpts_or_not
   Args:
       download_ckpts_or_not: bool variable. If true, then download the ckpts for ucsd1/ucsd2
   ```
   
2. Prepare the dataset:
    ```bash
    ./prepare_data.sh dataset datapath
    Args:
        dataset: UCSDped1, UCSDped2, Avenue
        datapath: the directory that you want to save the data, e.g., /tmp/anomaly_data/
    ```

#### Evaluate and Train the model

1. Evaluate the performance:
    ```bash
    ./run.sh ops dataset version ckptdownload datadir
    Args:
        ops: train, test, fps
        dataset: ucsd1, ucsd2, avenue
        version: int, experiment version, default: 0
        ckptdownload: bool variable. If true, evaluate the performance of the downloaded checkpoint. 
        datadir: the directory that you have saved your data, e.g., /tmp/anomaly_data/
    Example:
    ./run.sh test ucsd2 0 true /tmp/anomaly_data/ 
    ./run.sh fps ucsd2 0 true /tmp/anomaly_data/ 
    ```

2. Train the model:
    ```bash
    ./run.sh train ucsd1 0 false /tmp/anomaly_data/
    ```
    
#### Citation
If you use this code for your research, please cite our paper:
```
@article{DBLP:journals/corr/abs-2011-05054,
  author    = {Bo Li and
               Sam Leroux and
               Pieter Simoens},
  title     = {Decoupled Appearance and Motion Learning for Efficient Anomaly Detection
               in Surveillance Video},
  journal   = {CoRR},
  volume    = {abs/2011.05054},
  year      = {2020},
}
```
    
    
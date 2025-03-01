# ModernTCN Revisited: A Reproducibility Study with Extended Benchmarks

This repository contains the code for our paper "ModernTCN Revisited: A Reproducibility Study with Extended Benchmarks", which evaluates the reproducibility of [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://openreview.net/forum?id=vpJMJerXHU#) (ICLR 2024 Spotlight).

## ModernTCN Block

**ModernTCN block design:** 
Images below are from the original ModernTCN paper.

|![image](fig/fig_block.png) | ![image](fig/fig_erf.png)
|:--:|:--:|
| *Figure 1. ModernTCN block design.* | *Figure 2. Visualization of ERF.* |

## Get Started

1. Install Python 3.7 and necessary dependencies.
```
pip install -r requirements.txt
```

2. Download data. You can obtain most datasets from [Times-series-library](https://github.com/thuml/Time-Series-Library). For Speech Commands and PhysioNet datasets, use the provided download scripts:
```
cd ./ModernTCN-classification-extension/scripts
sh download_speech.sh
sh download_physionet.sh
```

3. Long-term forecasting tasks.
```
cd ./ModernTCN-Long-term-forecasting
sh ./scripts/ETTh2.sh
```

4. Short-term forecasting tasks (extended).
```
cd ./ModernTCN-short-term-extension
sh ./scripts/ETTm1.sh
```

5. Imputation tasks.
```
cd ./ModernTCN-imputation
sh ./scripts/ETTh2.sh
```

6. Classification tasks (extended).
```
cd ./ModernTCN-classification-extension
sh ./scripts/physionet.sh
sh ./scripts/speech_commands.sh
```

7. Anomaly detection tasks.
```
cd ./ModernTCN-detection
sh ./scripts/SWaT.sh
```

8. ERF Visualization for Speech Commands MFCC.
```
cd ./ModernTCN-classification-extension/scripts
sh erf_mfcc.sh
```

**Note:** The official ModernTCN source code does not provide code to visualize ERF. We adapted our visualization code from [RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch) to analyze the ERF for Speech Commands MFCC features.

## Citation

Our paper is currently under review. If you find this repo useful, please cite the original ModernTCN paper:
```
@inproceedings{
donghao2024moderntcn,
title={Modern{TCN}: A Modern Pure Convolution Structure for General Time Series Analysis},
author={Luo donghao and wang xue},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=vpJMJerXHU}
}
```

## Acknowledgement

We appreciate the following github repos for their valuable code base or datasets:

https://github.com/luodhhh/ModernTCN
https://github.com/ts-kim/RevIN
https://github.com/PatchTST/PatchTST
https://github.com/thuml/Time-Series-Library
https://github.com/facebookresearch/ConvNeXt
https://github.com/MegEngine/RepLKNet

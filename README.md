# MetaBCI

## Welcome! 
MetaBCI is an open-source platform for non-invasive brain computer interface. The project of MetaBCI is led by Prof. Minpeng Xu from Tianjin University, China. MetaBCI has 3 main parts:
* brainda: for importing dataset, pre-processing EEG data and implementing EEG decoding algorithms.
* brainflow: a high speed EEG online data processing framework.
* brainstim: a simple and efficient BCI experiment paradigms design module. 

This is the first release of MetaBCI, our team will continue to maintain the repository. If you need the handbook of this repository, please contact us by sending email to TBC_TJU_2022@163.com with the following information:
* Name of your teamleader
* Name of your university(or organization)

We will send you a copy of the handbook as soon as we receive your information.

## Sleep Staging — MetaBCI 创新应用开发赛项

### 项目简介

基于 MetaBCI 的单通道轻量级便携式睡眠监测系统。使用单通道 Fpz-Cz EEG 信号，30 秒为一 epoch，通过连续 3 个 epoch 构成时序上下文输入，实现 5 分类（W/N1/N2/N3/REM）睡眠分期。

**队伍：** 晓途队 | **单位：** 湘潭大学 | **赛道：** 被动监测

### 新增模块

| 模块 | 路径 | 功能 |
|------|------|------|
| SleepEDFDataset | `metabci/brainda/datasets/sleep_edf.py` | Sleep-EDF Expanded 数据集加载（153 人） |
| SleepParadigm | `metabci/brainda/paradigms/sleep.py` | 睡眠分期范式：center/causal 上下文窗口，5/4/3 分类标签映射 |
| ParaSleep | `metabci/brainda/algorithms/deep_learning/parasleep.py` | 132K 参数轻量模型：双分支深度可分离卷积 + patch-based MHA + 可选 Temporal Attention |
| SleepOnlineWorker | `metabci/brainflow/sleep_worker.py` | 在线推理器：因果滤波 + 信号质量门控 + LSL 推送 |
| EDFSleepPlayer | `metabci/brainflow/edf_player.py` | EDF 文件倍速回放，模拟在线数据流 |
| SleepMonitorUI | `metabci/brainstim/sleep_monitor.py` | 临床级睡眠报告：hypnogram + 阶段统计 |

### 快速开始

```bash
# 环境
conda create -n metabci python=3.9
pip install torch numpy scipy scikit-learn mne skorch onnx onnxruntime matplotlib pylsl

# 训练
python examples/sleep_staging/train_server.py --context 3 --save parasleep.pth --cache F:\sleep_cache

# 指标
python examples/sleep_staging/demo_metric.py --model parasleep.pth --split parasleep_split.npz --cache F:\sleep_cache --context 3

# 演示
python examples/sleep_staging/demo_e2e.py

# ONNX 导出
python examples/sleep_staging/export_onnx.py --checkpoint parasleep.pth --context 3 --cache F:\sleep_cache
```

### 当前结果（holdout test，10 人）

| 指标 | 值 |
|------|-----|
| Accuracy | 88.27% |
| Macro F1 | 0.7465 |
| Weighted F1 | 0.8920 |
| Cohen's Kappa | 0.7705 |
| W F1 | 0.9747 |
| N1 F1 | 0.4737 |
| N2 F1 | 0.7759 |
| N3 F1 | 0.7487 |
| REM F1 | 0.7594 |

配置：`ctx=3 center | wd=1e-2 | label_smoothing=0 | FocalLoss(γ=2) | 80 train / 10 test`

### 实验配置

```bash
# Baseline
python -u train_server.py --context 1 --save exp_ctx1.pth --cache F:\sleep_cache

# 核心消融
python -u train_server.py --context 3 --save exp_ctx3_center.pth --cache F:\sleep_cache
python -u train_server.py --context 3 --causal --save exp_ctx3_causal.pth --cache F:\sleep_cache

# N1 优化
python -u train_server.py --context 3 --sampler weighted --save exp_weighted.pth --cache F:\sleep_cache

# Temporal Attention
python -u train_server.py --model ta --context 3 --save exp_ta.pth --cache F:\sleep_cache

# 5-fold CV
python -u train_server.py --context 3 --cv 5 --save final_cv.pth --cache F:\sleep_cache
```

---

## Paper

If you find MetaBCI useful in your research, please cite:

Mei, J., Luo, R., Xu, L., Zhao, W., Wen, S., Wang, K., ... & Ming, D. (2023). MetaBCI: An open-source platform for brain-computer interfaces. Computers in Biology and Medicine, 107806.

And this open access paper can be found here: [MetaBCI](https://www.sciencedirect.com/science/article/pii/S0010482523012714)

## Content

- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [Paper](#paper)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [What do we need?](#what-do-we-need)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* BCI datasets come in different formats and standards
* It's tedious to figure out the details of the data
* Lack of python implementations of modern decoding algorithms
* It's not an easy thing to perform BCI experiments especially for the online ones.

If someone new to the BCI wants to do some interesting research, most of their time would be spent on preprocessing the data, reproducing the algorithm in the paper, and also find it difficult to bring the algorithms into BCI experiments.

### The solution

The Meta-BCI will:

* Allow users to load the data easily without knowing the details
* Provide flexible hook functions to control the preprocessing flow
* Provide the latest decoding algorithms
* Provide the experiment UI for different paradigms (e.g. MI, P300 and SSVEP)
* Provide the online data acquiring pipeline.
* Allow users to bring their pre-trained models to the online decoding pipeline.

The goal of the Meta-BCI is to make researchers focus on improving their own BCI algorithms and performing their experiments without wasting too much time on preliminary preparations.

## Features

* Improvements to MOABB APIs
   - add hook functions to control the preprocessing flow more easily
   - use joblib to accelerate the data loading
   - add proxy options for network connection issues
   - add more information in the meta of data
   - other small changes

* Supported Datasets
   - MI Datasets
     - AlexMI
     - BNCI2014001, BNCI2014004
     - PhysionetMI, PhysionetME
     - Cho2017
     - MunichMI
     - Schirrmeister2017
     - Weibo2014
     - Zhou2016
   - SSVEP Datasets
     - Nakanishi2015
     - Wang2016
     - BETA

* Implemented BCI algorithms
   - Decomposition Methods
     - SPoC, CSP, MultiCSP and FBCSP
     - CCA, itCCA, MsCCA, ExtendCCA, ttCCA, MsetCCA, MsetCCA-R, TRCA, TRCA-R, SSCOR and TDCA
     - DSP
   - Manifold Learning
     - Basic Riemannian Geometry operations
     - Alignment methods
     - Riemann Procustes Analysis
   - Deep Learning
     - ShallowConvNet
     - EEGNet
     - ConvCA
     - GuneyNet
     - Cross dataset transfer learning based on pre-training
   - Transfer Learning
     - MEKT
     - LST

## Installation

### Quick Install (Recommended)

Install MetaBCI with all features:
```sh
pip install metabci[all]
```

### Modular Installation

MetaBCI supports modular installation - install only what you need:

```sh
# Core only (minimal, for custom setups)
pip install metabci

# brainda: datasets, algorithms, deep learning
pip install metabci[brainda]

# brainflow: signal acquisition (lightweight)
pip install metabci[brainflow]

# brainstim: stimulus presentation
pip install metabci[brainstim]

# Combine modules as needed
pip install metabci[brainda,brainflow]
```

### Development Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TBC-TJU/MetaBCI.git
   cd MetaBCI
   ```

2. Install in development mode with all dependencies
   ```sh
   pip install -e .[all,dev,docs]
   ```

   Or using requirements files:
   ```sh
   pip install -r requirements-dev.txt
   pip install -e .
   ```

### Conda Installation

For conda users, an environment file is provided:
```sh
conda env create -f environment.yml
conda activate metabci
```

### Using uv (Fast Alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:
```sh
uv pip install metabci[all]
```
## Who are we?

The MetaBCI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China
- Tianjin Brain Center, China


## What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation and technical writing and project management.

We'd love your feedback along the way.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: TBC_TJU_2022@163.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)

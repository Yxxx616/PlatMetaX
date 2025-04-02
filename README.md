# PlatMetaX
**A Matlab platform for meta-black-box optimization, covering rl-based, sl-based, el-based meta-learning.**
![PlatMetaX Logo](https://github.com/Yxxx616/PlatMetaX/blob/main/GUI/platmetaxLOGO2.0.png)

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)
![MATLAB Version](https://img.shields.io/badge/MATLAB-R2021a%2B-orange)
![Release](https://img.shields.io/badge/release-2.0-success)


## Paper reference
https://doi.org/10.48550/arXiv.2503.22722


## Key Features

### 🚀 Robust MATLAB Foundation
- Built on MATLAB® for seamless integration with scientific computing workflows  
- Native support for reinforcement learning (RL) and neural network toolboxes  
- Zero-configuration setup for rapid meta-optimizer development

### 📊 Standardized MetaBBO Workflows
- Unified framework for three meta-learning paradigms:  
  - **RL-based** (Reinforcement Learning)  
  - **SL-based** (Supervised Learning)  
  - **EL-based** (Evolutionary Learning)  
- Parameter-free operation with adaptive configuration management

### 🧠 Advanced Meta-Optimization Modules
- Prebuilt integration with large language models (LLMs) via Python-MATLAB API bridge  
- Transformer-based meta-optimizer implementations  
- Expandable architecture for custom algorithm integration

### 📚 Comprehensive Benchmarking Suite
- Curated collection of 50+ traditional optimization algorithms  
- 150+ benchmark problems from BBOB, CEC, LIRCMOP, and TSPLIB  
- Built-in performance comparison tools

### 🖥️ Intuitive Graphical Interface
- Visual experiment configuration  
- Real-time optimization tracking  
- Automated report generation

---

## Version History

### v2.0 (April 2025)
- Introduced LLM-based meta-optimizers via MATLAB-Python API integration  
- Enhanced cross-version compatibility for RL optimizers  
- Added Transformer-based meta-optimizer (`Transformer_DE_Sol_Metaoptimizer.m`)  
- Expanded benchmark problem sets

### v1.0 (January 2024)
- Core framework implementation  
- Baseline RL/SL/EL meta-optimizers  
- Basic GUI functionality

---

## Getting Started

### Prerequisites
- MATLAB R2021a or later (2024a  for transformer integration)
- Python 3.8+ and openai (for LLM integration)  

### Installation
```bash
git clone https://github.com/Yxxx616/PlatMetaX.git
addpath(genpath('PlatMetaX'));
```

### Basic Usage
```matlab
% Train a meta-optimizer
platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet', 'BBOB', 'N', 50, 'D', 10);

% Test meta-optimizer performance
platmetabbo('task', @Test, 'metabboComps', 'DDPG_DE_F', 'problemSet', 'CEC2020');
```

### GUI Launch
```matlab
platmetax;
```

## Development Guidelines
### Custom Meta-Optimizer Implementation
1.Define base optimizer components
2.Parameterize target optimization aspects
3.Design state-action space in Environment.m
4.Configure observation/action specs in observationInfo/actionInfo

### Experimental Configuration
1.Training parameters: Modify Train.m
2.Testing parameters: Adjust Test.m
3.Dataset splitting: Configure Utils/splitProblemSet.m

## Citation & Licensing
```bibtex
@misc{platmetax2025,
  title={PlatMetaX: A MATLAB Platform for Meta-Black-Box Optimization},
  author={Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He},
  year={2025},
  doi={10.48550/arXiv.2503.22722}
}
```
- Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX for research purposes. All publications which use this platform or MetaBBO code in the platform should acknowledge the use of "PlatMetaX" and reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. PlatMetaX: A MATLAB platform for meta-black-box optimization. https://doi.org/10.48550/arXiv.2503.22722".
- License: Academic use only. Commercial applications require written permission.
- Dependency Notice: Built upon PlatEMO framework (Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for research purposes. All publications which use any code from PlatEMO in the platform should acknowledge the use of "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87".)

## Community & Support
### 📬 Contact
- Lead Developer: Dr. Xu Yang
- Email: 501216619@qq.com
- Discussion Group:

![QQ Group QR Code](https://github.com/Yxxx616/PlatMetaX/blob/main/GUI/qqmobCode.png "Join Our QQ Group")



# PlatMetaX
**A Matlab platform for meta-black-box optimization, covering rl-based, sl-based, ec-based meta-learning.**
![PlatMetaX Logo](https://github.com/Yxxx616/PlatMetaX/blob/main/GUI/platmetaxLOGO2.0.png)

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)
![MATLAB Version](https://img.shields.io/badge/MATLAB-R2021a%2B-orange)
![Release](https://img.shields.io/badge/release-2.0-success)


## Paper reference
https://doi.org/10.48550/arXiv.2503.22722


## Features

### 1. 强大的 MATLAB 基础
PlatMetaX 基于 MATLAB 平台开发，充分利用了其在科学计算和用户友好性方面的优势。无需复杂配置，用户即可轻松集成强化学习和神经网络工具，快速开发元优化器。

### 2. 全面的 MetaBBO 方法和标准化的工作流程
PlatMetaX 提供了基于强化学习、基于监督学习和基于进化学习的元优化器模块用于元学习，用户无需调整参数即可直接使用。此外，PlatMetaX 标准化了这些不同类型元优化器的工作流程，用户可以在平台上快速开发和测试各类型的元黑箱优化算法，极大地提高了开发效率。

### 3. 丰富的传统算法库和基准测试问题库
PlatMetaX 提供了丰富的传统优化算法和大量的基准测试优化问题，用户可以轻松设计基础优化器，并开展 MetaBBO 算法与传统算法的对比试验研究。

### 4. 直观的图形用户界面 GUI
PlatMetaX 配备了图形用户界面，使得实验测试和结果可视化更加便捷，提升了用户体验。

## Version Introduction
# V1.0
- 实现了RL-,SL-,EL-based meta-optimizers
- GUI只能用于测试。

# V2.0
- EL-based meta-optimizers引入基于LLM的元优化器,通过【Matlab调用python->python调用大模型API】实现。（需要配置python环境）
- 修改了RL-based optimizer的实例化逻辑，使其可以在不同MATLAB版本中运行。
- 增加了transformer模型作为元优化器（详见Transformer_DE_Sol_Metaoptimizer.m）


## Quick Start

# Installation
```bash
git clone https://github.com/Yxxx616/PlatMetaX.git
```

# Main function
platmetax.m

# Some exanmples
1. 训练一个meta-optimizer
- platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB')
- platmetax('task', @Train, 'metabboComps', 'DQN_DE_MS', 'problemSet','BBOB')
- platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB','N',50,'D',10)
- platmetax('task', @Train, 'metabboComps', 'DE_DE_FCR', 'problemSet','BBOBEC','N',50,'D',10)
2. 测试训练好的meta-optimizer
- （1）测试RL-based meta-optimizer：DDPG_DE_F（实现使用DDPG在线调整DE参数F）
-  platmetabbo('task', @Test, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB') 
-  测试RL-based meta-optimizer：DQN_DE_MS（实现使用DQN在线调整DE的变异策略MS）
-  platmetabbo('task', @Test, 'metabboComps', 'DQN_DE_MS', 'problemSet','BBOB')
- （2）测试SL-based meta-optimizer：MLP_Alg_Rec（实现使用MLP神经网络对不同TSP示例算法推荐ABC/CSO/DE/PSO/SA）
-  platmetabbo('task', @Test, 'metabboComps', 'MLP_Alg_Rec', 'problemSet','TSPs')
- （3）测试EC-based meta-optimizer：DE_DE_FCR（实现使用DE离线调整DE参数F和CR）
- platmetabbo('task', @Test, 'metabboComps', 'DE_DE_FCR', 'problemSet','BBOB')

## NOTE
1. 写自己的MetaBBO时需要先定义base-optimizer，思考参数化哪部分（学习base-optimizer的什么东西），然后设计metaoptimizer的输入也就是state，然后根据state的大小在Environment中定义observationInfo和actionInfo。
2. 训练的参数在Train.m中修改，测试的参数在Test.m中修改。训练集和测试集的切分在Utils下的splitProblemSet函数中修改。如果想在各种各样的问题集上进行训练，可以新建一个问题集名字，把你想用于训练的问题都放进去，然后统一一个命名规范即可。
3. 使用命令行测试时只可以测试一个算法，但是测试问题可以通过修改'problemSet'参数的值为'LIRCMOP'、'CF'等platEMO包含的任何测试问题集，只需要测试问题集的名字即可。
4. 测试时建议选择用GUI，直接运行platmetax.m，进入到GUI界面，选择test模块或exp模块，再选择自己训练好的base-optimizer(点击标签“learned”可以快速找到这些学习型算法)，test可以测试单独函数，exp可以测试多个函数并且可以和任何platemo里想对比的算法进行对比试验！！

# PlatMetaX Copyright
Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX for research purposes. All publications which use this platform or MetaBBO code in the platform should acknowledge the use of "PlatMetaX" and reference "Xu Yang, and Rui Wang. PlatMetaX: A MATLAB platform for meta-black-box optimization, https://doi.org/10.48550/arXiv.2503.22722".


# Contact Us

## Join Our Community
- **QQ Group**: 826619039  
![QQ Group QR Code](https://github.com/Yxxx616/PlatMetaX/blob/main/GUI/qqmobCode.png "Join Our QQ Group")

## Contact Us via Email
- **Email Address**: 501216619@qq.com

### Note on Link Parsing
The image link above may fail to parse due to network issues or an invalid URL. If you encounter this problem, please check the legality of the link and try again later. If you do not require parsing of this specific link, feel free to continue with other inquiries.

- *声明：非完全原创，基于platEMO平台*
- *建议：有PlatEMO基础的同学食用更佳*
### PlatEMO Copyright
Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for research purposes. All publications which use any code from PlatEMO in the platform should acknowledge the use of "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87".


# PlatMetaX
**A Matlab platform for meta-black-box optimization, covering rl-assisted EA meta-learning.**
- *声明：非完全原创，基于platEMO平台*
- *建议：有PlatEMO基础的同学食用更佳*

# V1.0
- 实现了RL-,SL-,EL-based meta-optimizers
- GUI只能用于测试。

# Main function
platmetax.m

# Quick Start
1. 训练一个meta-optimizer
- platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB')
- platmetax('task', @Train, 'metabboComps', 'DQN_DE_MS', 'problemSet','BBOB')
- platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB','N',50,'D',10)
- platmetax('task', @Train, 'metabboComps', 'DE_DE_FCR', 'problemSet','BBOBEC','N',50,'D',10)
2. 测试训练好的meta-optimizer
%   （1）测试RL-based meta-optimizer：DDPG_DE_F（实现使用DDPG在线调整DE参数F）
%   platmetabbo('task', @Test, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB') 
%   测试RL-based meta-optimizer：DQN_DE_MS（实现使用DQN在线调整DE的变异策略MS）
%   platmetabbo('task', @Test, 'metabboComps', 'DQN_DE_MS', 'problemSet','BBOB')
%   （2）测试SL-based meta-optimizer：MLP_Alg_Rec（实现使用MLP神经网络对不同TSP示例算法推荐ABC/CSO/DE/PSO/SA）
%   platmetabbo('task', @Test, 'metabboComps', 'MLP_Alg_Rec', 'problemSet','TSPs')
%   （3）测试EC-based meta-optimizer：DE_DE_FCR（实现使用DE离线调整DE参数F和CR）
%   platmetabbo('task', @Test, 'metabboComps', 'DE_DE_FCR', 'problemSet','BBOB')

# NOTE
1. 写自己的MetaBBO时需要先定义base-optimizer，思考参数化哪部分（学习base-optimizer的什么东西），然后设计metaoptimizer的输入也就是state，然后根据state的大小在Environment中定义observationInfo和actionInfo。
2. 训练的参数在Train.m中修改，测试的参数在Test.m中修改。训练集和测试集的切分在Utils下的splitProblemSet函数中修改。如果想在各种各样的问题集上进行训练，可以新建一个问题集名字，把你想用于训练的问题都放进去，然后统一一个命名规范即可。
3. 使用命令行测试时只可以测试一个算法，但是测试问题可以通过修改'problemSet'参数的值为'LIRCMOP'、'CF'等platEMO包含的任何测试问题集，只需要测试问题集的名字即可。
4. 测试时建议选择用GUI，直接运行platmetax.m，进入到GUI界面，选择test模块或exp模块，再选择自己训练好的base-optimizer(点击标签“learned”可以快速找到这些学习型算法)，test可以测试单独函数，exp可以测试多个函数并且可以和任何platemo里想对比的算法进行对比试验！！

# Conclusion
暂且先这样啦，后续再更新说明文档
# FIGHTING!

# PlatMetaX Copyright
Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX for research purposes. All publications which use this platform or MetaBBO code in the platform should acknowledge the use of "PlatMetaX" and reference "Xu Yang, and Rui Wang. PlatMetaX: A MATLAB platform for meta-black-box optimization, *畅想一下NIPS, 2025, .(.): ..-..*".

# PlatEMO Copyright
Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for research purposes. All publications which use any code from PlatEMO in the platform should acknowledge the use of "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87".

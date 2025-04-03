classdef Transformer_DE_UpEq_Metaoptimizer < rl.agent.CustomAgent
% to do %
% Algorithm generation-update equation generation
% required MATLAB >= 2024a
% Transformer neural network as meta-policy, being trained via RL.
% DE as base-optimizer, with update equation as opotimization object.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        % 核心组件
        ActorNetwork
        TargetNetwork
        xExperienceBuffer = struct(...
            'Observations', {}, ...
            'Actions', {}, ...
            'Rewards', {}, ...
            'NextObservations', {}, ...
            'IsDone', {})
        BufferMaxSize = 2000;
        
        % 优化器状态
        Velocity
        SquaredGradient
        
        % 超参数
        Gamma = 0.95                  % 折扣因子
        LearnRate = 1e-4              % 学习率
        TargetUpdateFrequency = 100   % 目标网络更新频率
        MiniBatchSize = 32            % 训练批大小
        ExplorationEpsilon = 0.1      % 探索率
        GradientDecayFactor = 0.9     % 梯度衰减因子
        SquaredGradientDecayFactor = 0.999 % 平方梯度衰减因子
        
        % 状态跟踪
        TrainingStep = 0
        MaxTreeDepth = 3
    end
    
    methods
        function obj = Transformer_DE_UpEq_Metaoptimizer(obsInfo, actInfo)
            % 调用父类构造函数
            obj = obj@rl.agent.CustomAgent();
            obj.ObservationInfo = obsInfo;
            obj.ActionInfo = actInfo;  
            % 初始化网络架构
            obj.ActorNetwork = obj.createTransformerNetwork(obsInfo, actInfo);
            obj.TargetNetwork = obj.ActorNetwork;
            
            % 初始化优化器状态
            obj.Velocity = [];
            obj.SquaredGradient = [];
        end
    end
    methods (Access = protected)
        function action = getActionImpl(obj, observation)
            % 确定性动作选择
            processedObs = obj.preprocessObservation(observation);
            actionProbs = predict(obj.ActorNetwork, processedObs);
            action = obj.sampleAction(actionProbs, 0); % 无探索
        end
        
        function action = getActionWithExplorationImpl(obj, observation)
            % 带探索的动作选择
            processedObs = obj.preprocessObservation(observation);
            actionProbs = predict(obj.ActorNetwork, processedObs);
            action = obj.sampleAction(actionProbs, obj.ExplorationEpsilon);
        end
        
        function learnImpl(obj, experience)
            % 存储经验并执行训练
            if numel(obj.xExperienceBuffer) > obj.BufferMaxSize
                obj.xExperienceBuffer(randi(obj.BufferMaxSize)) = experience;
            else
                obj.xExperienceBuffer(end+1) = experience;
            end
            
            if numel(obj.xExperienceBuffer) >= obj.MiniBatchSize
                % 从经验池采样
                miniBatch = randperm(numel(obj.xExperienceBuffer), obj.MiniBatchSize);
                
                % 转换数据格式
                [states, actions, rewards, nextStates, dones] = ...
                    obj.unpackExperience(miniBatch);
                
                % 计算目标Q值
                targetQ = obj.computeTargetQ(rewards, nextStates, dones);
                
                % 计算梯度并更新网络
                [gradients, loss] = dlfeval(...
                    @obj.computeGradients, states, actions, targetQ);
                
                % 使用Adam更新参数
                [obj.ActorNetwork, obj.Velocity, obj.SquaredGradient] = adamupdate(...
                    obj.ActorNetwork, gradients, ...
                    obj.Velocity, obj.SquaredGradient, ...
                    obj.TrainingStep, ...
                    obj.LearnRate, ...
                    obj.GradientDecayFactor, ...
                    obj.SquaredGradientDecayFactor);
                
                % 定期更新目标网络
                if mod(obj.TrainingStep, obj.TargetUpdateFrequency) == 0
                    obj.TargetNetwork = copy(obj.ActorNetwork);
                end
                obj.TrainingStep = obj.TrainingStep + 1;
            end
        end
        
        function resetImpl(obj)
            % 重置目标网络和训练状态
            obj.TargetNetwork = copy(obj.ActorNetwork);
            obj.TrainingStep = 0;
        end
    end
    
    methods (Access = private)
        function net = createTransformerNetwork(obj, obsInfo,actInfo)
            % 构建Transformer网络架构
            inputDim = obsInfo.Dimension(2);  
            outputDim = actInfo.Dimension(1);
                        
            layers = [
                sequenceInputLayer(inputDim, 'Name', 'input')  % 输入层
                
                % 编码器部分 (重复两次)
                % 编码器块1
                selfAttentionLayer(8, 256, 'Name', 'enc_self_attn1')  % 维度保持256
                additionLayer(2, 'Name', 'enc_add1')                  
                layerNormalizationLayer('Name', 'enc_ln1')            
                fullyConnectedLayer(256*2, 'Name', 'enc_fc1')         % 扩展维度到512
                reluLayer('Name', 'enc_relu1')                        
                fullyConnectedLayer(inputDim, 'Name', 'enc_fc2')           % 恢复维度到inputDim
                additionLayer(2, 'Name', 'enc_add2')                  
                layerNormalizationLayer('Name', 'enc_ln2')            
                
                % 编码器块2 (重复结构)
                selfAttentionLayer(8, 256, 'Name', 'enc_self_attn2')  % 维度保持256
                additionLayer(2, 'Name', 'enc_add3')
                layerNormalizationLayer('Name', 'enc_ln3')
                fullyConnectedLayer(256*2, 'Name', 'enc_fc3')
                reluLayer('Name', 'enc_relu2')
                fullyConnectedLayer(inputDim, 'Name', 'enc_fc4')          % 恢复维度到256
                additionLayer(2, 'Name', 'enc_add4')
                layerNormalizationLayer('Name', 'enc_ln4')
                
                % 解码器部分
                % 解码器自注意力
                selfAttentionLayer(8, 256, 'AttentionMask', 'causal', 'Name', 'dec_self_attn')
                additionLayer(2, 'Name', 'dec_add1')
                layerNormalizationLayer('Name', 'dec_ln1')
                
                % 交叉注意力（编码器->解码器）
                attentionLayer(4, 'Name', 'dec_cross_attn')  
                layerNormalizationLayer('Name', 'dec_ln2')
                
                % 前馈网络（带维度匹配）
                fullyConnectedLayer(256*2, 'Name', 'dec_fc1')
                reluLayer('Name', 'dec_relu1')
                fullyConnectedLayer(inputDim, 'Name', 'dec_fc2')          % 恢复维度到256
                additionLayer(2, 'Name', 'dec_add3')
                layerNormalizationLayer('Name', 'dec_ln3')
                
                % 输出部分（新增完整输出结构）
                fullyConnectedLayer(256, 'Name', 'output_fc1')        % 过渡层
                reluLayer('Name', 'output_relu')
                fullyConnectedLayer(outputDim, 'Name', 'final_fc')
                softmaxLayer('Name', 'prob_output')                   % 输出层
            ];
            
            % 创建 layerGraph 并连接
            lgraph = layerGraph(layers);
            
            %% 编码器
            lgraph = connectLayers(lgraph, 'input', 'enc_add1/in2');   % 跳跃连接输入
            
            % 第一层前馈残差（确保维度匹配）
            lgraph = connectLayers(lgraph, 'enc_ln1', 'enc_add2/in2');
            
            % 第二层自注意力残差
            lgraph = connectLayers(lgraph, 'enc_ln2', 'enc_add3/in2');
            
            % 第二层前馈残差
            lgraph = connectLayers(lgraph, 'enc_ln3', 'enc_add4/in2');
            
            %% 解码器连接修正
            % 自注意力残差
            lgraph = connectLayers(lgraph, 'input', 'dec_add1/in2');  % 假设共享输入维度
            
            % 交叉注意力连接
            lgraph = connectLayers(lgraph, 'enc_ln4', 'dec_cross_attn/key');
            lgraph = connectLayers(lgraph, 'enc_ln4', 'dec_cross_attn/value');
            
            % 前馈残差连接
            lgraph = connectLayers(lgraph, 'dec_ln2', 'dec_add3/in2');
            
            % analyzeNetwork(lgraph)
            % 创建可训练网络
            net = dlnetwork(lgraph);
        end
        
        function action = sampleAction(obj, actionProbs, epsilon)
            % 带epsilon探索的动作采样
            if rand < epsilon
                % 随机探索
                action = randi([0 1], obj.MaxTreeDepth*2, 1);
            else
                % 按概率采样
                sampled = rand(size(actionProbs)) < actionProbs;
                action = double(gather(extractdata(sampled)));
            end
        end
        
        function [gradients, loss] = computeGradients(obj, states, actions, targetQ)
            % 自定义梯度计算函数
            [predictions, state] = forward(obj.ActorNetwork, states);
            
            % 计算二元交叉熵损失
            loss = crossentropy(...
                predictions, actions, ...
                'TargetCategories', 'independent', ...
                'DataFormat', 'CB');
            
            % 计算梯度
            gradients = dlgradient(loss, obj.ActorNetwork.Learnables);
        end
        
        function targetQ = computeTargetQ(obj, rewards, nextStates, dones)
            % 双重Q学习目标计算
            currentQ = forward(obj.ActorNetwork, nextStates);
            nextQ = forward(obj.TargetNetwork, nextStates);
            
            targetQ = rewards + obj.Gamma .* (1 - dones) .* max(nextQ, [], 1);
        end
        
        function processedObs = preprocessObservation(obj, observation)
            % 观测数据预处理
            processedObs = dlarray(observation{1}, 'TC');  % Time x Channels
        end
        
        function [states, actions, rewards, nextStates, dones] = unpackExperience(~, batch)
            % 解包经验数据
            states = batch{1};
            actions = batch{2};
            rewards = batch{3};
            nextStates = batch{4};
            dones = batch{5};
        end
    end
    
    methods (Static)
        function transitions = sampleTransitions(experiences, indices)
            % 自定义经验采样方法
            transitions = cell(5,1);
            for i = 1:numel(indices)
                transition = experiences(indices(i));
                transitions{1}{i} = transition.Observation;
                transitions{2}{i} = transition.Action;
                transitions{3}{i} = transition.Reward;
                transitions{4}{i} = transition.NextObservation;
                transitions{5}{i} = transition.IsDone;
            end
        end
    end
end
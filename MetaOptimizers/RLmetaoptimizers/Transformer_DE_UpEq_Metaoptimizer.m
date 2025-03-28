classdef Transformer_DE_UpEq_Metaoptimizer < rl.agent.CustomAgent
    properties
        % 核心组件
        ActorNetwork
        TargetNetwork
        Optimizer
        ExperienceBuffer
        
        % 超参数
        Gamma = 0.95                  % 折扣因子
        LearnRate = 1e-4              % 学习率
        TargetUpdateFrequency = 100   % 目标网络更新频率
        MiniBatchSize = 32            % 训练批大小
        ExplorationEpsilon = 0.1      % 探索率
        
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
            obj.ActorNetwork = obj.createTransformerNetwork(obsInfo, obj.MaxTreeDepth);
            obj.TargetNetwork = copy(obj.ActorNetwork);
            
            % 配置优化器
            obj.Optimizer = adamOptimizer(...
                'LearnRateSchedule', 'piecewise',...
                'LearnRate', obj.LearnRate,...
                'GradientDecayFactor', 0.9,...
                'SquaredGradientDecayFactor', 0.999);
            
            % 初始化经验池
            obj.ExperienceBuffer = rl.replay.Buffer(...
                obsInfo, actInfo, 1e5,...
                'SampleTransitionsFunction', @this.sampleTransitions);
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
            append(obj.ExperienceBuffer, experience);
            
            if length(obj.ExperienceBuffer) >= obj.MiniBatchSize
                % 从经验池采样
                miniBatch = sample(obj.ExperienceBuffer, obj.MiniBatchSize);
                
                % 转换数据格式
                [states, actions, rewards, nextStates, dones] = ...
                    obj.unpackExperience(miniBatch);
                
                % 计算目标Q值
                targetQ = obj.computeTargetQ(rewards, nextStates, dones);
                
                % 计算梯度并更新网络
                [gradients, loss] = dlfeval(...
                    @obj.computeGradients, states, actions, targetQ);
                
                obj.ActorNetwork = adamupdate(...
                    obj.ActorNetwork, gradients, obj.Optimizer);
                
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
        function net = createTransformerNetwork(obj, obsInfo, maxTreeDepth)
            % 构建Transformer网络架构
            inputDim = obsInfo.Dimension(2);  % (D + M + 1)
            outputDim = maxTreeDepth * 2;
            
            layers = [
                sequenceInputLayer(inputDim, 'Name', 'input')
                
                % 位置嵌入层
                positionEmbeddingLayer(64, 128, 'Name', 'pos_embed')
                
                % 编码器模块
                additionLayer(2, 'Name', 'add1')
                selfAttentionLayer(8,256)
                selfAttentionLayer(8,256)
                layerNormalizationLayer('Name', 'ln_embed')
                fullyConnectedLayer(256, 'Name', 'fc1')
                layerNormalizationLayer('Name', 'ln2')
                
                % 解码器模块
                selfAttentionLayer(8,256,'AttentionMask','causal')
                additionLayer(2, 'Name', 'add2')
                layerNormalizationLayer('Name', 'ln_embed')
                selfAttentionLayer(8,256)
                additionLayer(2, 'Name', 'add3')
                layerNormalizationLayer('Name', 'ln_embed')
                fullyConnectedLayer(256, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                layerNormalizationLayer('Name', 'ln1')
                
                fullyConnectedLayer(128, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                layerNormalizationLayer('Name', 'ln2')
                
                fullyConnectedLayer(outputDim, 'Name', 'output')
                sigmoidLayer('Name', 'prob_output')
            ];
            
            net = dlnetwork(layers);
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
            processedObs = dlarray(observation, 'CTB');  % Channel x Time x Batch
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
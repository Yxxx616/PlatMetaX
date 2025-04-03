classdef Transformer_DE_Sol_Metaoptimizer < rl.agent.CustomAgent
% Algorithm generation-solution manipulation
% required MATLAB >= 2024a
% Transformer neural network as meta-policy, being trained via RL.
% DE as base-optimizer, with solution manipulation as opotimization object.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: A MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        % 核心组件
        ActorNetwork
        CriticNetwork
        TargetActorNetwork
        TargetCriticNetwork

        xExperienceBuffer = table('Size', [0 5], 'VariableTypes', ...
            {'cell', 'cell', 'double', 'cell', 'uint8'}, ...
            'VariableNames', {'Observations', 'Actions', 'Rewards', 'NextObservations', 'IsDone'});
        BufferMaxSize = 2000;
        
        % 优化器状态
        ActorVelocity
        ActorSquaredGradient
        CriticVelocity
        CriticSquaredGradient
        
        % 超参数
        Gamma = 0.95                  % 折扣因子
        LearnRate = 1e-4              % 学习率
        TargetUpdateFrequency = 100   % 目标网络更新频率
        MiniBatchSize = 32            % 训练批大小
        ExplorationEpsilon = 0.1      % 探索率
        GradientDecayFactor = 0.9     % 梯度衰减因子
        SquaredGradientDecayFactor = 0.999 % 平方梯度衰减因子
        
        % 状态跟踪
        TrainingStep = 1
        MaxTreeDepth = 3
    end
    
    methods
        function obj = Transformer_DE_Sol_Metaoptimizer(obsInfo, actInfo)
            % 调用父类构造函数
            obj = obj@rl.agent.CustomAgent();
            obj.ObservationInfo = obsInfo;
            obj.ActionInfo = actInfo;  
            % 初始化网络架构
            obj.ActorNetwork = obj.createTransformerNetwork(obsInfo, actInfo);
            obj.TargetActorNetwork = obj.ActorNetwork;
            obj.CriticNetwork = obj.createCriticNetwork(obsInfo, actInfo);
            obj.TargetCriticNetwork = obj.CriticNetwork;

            % 初始化优化器状态
            obj.ActorVelocity = [];
            obj.ActorSquaredGradient = [];
            obj.CriticVelocity = [];
            obj.CriticSquaredGradient = [];
        end
    end
    methods (Access = protected)
        function action = getActionImpl(obj, observation)
            actionProbs = predict(obj.ActorNetwork, observation{1});
            action = obj.sampleAction(actionProbs, 0); % 无探索
        end
        
        function action = getActionWithExplorationImpl(obj, observation)
            actionDLarray = predict(obj.ActorNetwork, observation{1});
            if rand < obj.ExplorationEpsilon
                action = rand(size(actionDLarray));
            else
                action = double(extractdata(actionDLarray));
            end
        end
        
        function learnImpl(obj, experience)
            % 存储经验并执行训练
            if height(obj.xExperienceBuffer) >= obj.BufferMaxSize
                % 随机替换一个经验
                idx = randi(obj.BufferMaxSize);
                obj.xExperienceBuffer(idx, :) = experience;
            else
                obj.xExperienceBuffer(end+1, :) = experience;
            end
            
            if height(obj.xExperienceBuffer) >= obj.MiniBatchSize
                % 从经验池采样
                
                miniBatch = obj.sampleMiniBatch();
                
                % 转换数据格式
                [states, actions, rewards, nextStates, dones] = ...
                    obj.unpackExperience(miniBatch);
                
                % ----------------- Critic网络更新 -----------------
                % 计算目标Q值（使用目标网络）
                nextActions = predict(obj.TargetActorNetwork, nextStates);
                targetQ = predict(obj.TargetCriticNetwork, nextStates, nextActions);
                targetQ = rewards + obj.Gamma * (1 - dones) .* extractdata(targetQ);
                
                [loss, criticGradients] = dlfeval(@obj.criticLossFunction, states, actions, targetQ, obj.CriticNetwork);
                                
                % 反向传播更新Critic
                [obj.CriticNetwork, obj.CriticVelocity, obj.CriticSquaredGradient] = ...
                    adamupdate(obj.CriticNetwork, criticGradients, ...
                    obj.CriticVelocity, obj.CriticSquaredGradient, obj.TrainingStep, obj.LearnRate, ...
                    obj.GradientDecayFactor, ...
                    obj.SquaredGradientDecayFactor);
                
                % ----------------- Actor网络更新 -----------------
 
                % 使用 dlfeval 计算梯度
                [loss, actorGradients] = dlfeval(@obj.actorLossFunction, states, obj.ActorNetwork, obj.CriticNetwork);
                
                [obj.ActorNetwork, obj.ActorVelocity, obj.ActorSquaredGradient] = ...
                    adamupdate(obj.ActorNetwork, actorGradients, ...
                    obj.ActorVelocity, obj.ActorSquaredGradient, obj.TrainingStep, obj.LearnRate, ...
                    obj.GradientDecayFactor, ...
                    obj.SquaredGradientDecayFactor);
                
                % ----------------- 目标网络同步 -----------------
                if mod(obj.TrainingStep, obj.TargetUpdateFrequency) == 0
                    obj.TargetActorNetwork = obj.ActorNetwork; % 硬更新目标网络
                    obj.TargetCriticNetwork = obj.CriticNetwork;
                end
                obj.TrainingStep = obj.TrainingStep + 1;
            end
        end
        function [loss, gradients] = criticLossFunction(obj, states, actions, targetQ, CriticNetwork)
            targetQ = dlarray(targetQ,'CBT');
            currentQ = predict(CriticNetwork, states, actions);
    
            % 计算Critic损失（均方误差）
            loss = mse(currentQ, targetQ);
            
            % 计算梯度
            gradients = dlgradient(loss, CriticNetwork.Learnables);
        end

        function [loss, gradients] = actorLossFunction(obj, states, ActorNetwork, CriticNetwork)
                    % 计算Actor策略的Q值（最大化期望回报）
            actorActions = predict(ActorNetwork, states);
            actorQ = predict(CriticNetwork, states, actorActions);
            loss = -mean(actorQ, 'all'); % 确保loss是一个标量
            gradients = dlgradient(loss, ActorNetwork.Learnables);
        end

        function miniBatch = sampleMiniBatch(obj)
            % 从经验池中随机采样
            indices = randperm(height(obj.xExperienceBuffer), obj.MiniBatchSize);
            miniBatch = obj.xExperienceBuffer(indices, :);
        end

        
        
        function resetImpl(obj)
            % 重置目标网络和训练状态
            obj.TargetNetwork = obj.ActorNetwork;
            obj.TrainingStep = 0;
        end
    end
    
    methods (Access = private)
        function net = createTransformerNetwork(obj, obsInfo,actInfo)
            % 构建Transformer网络架构
            inputDim = obsInfo.Dimension(1);  
            outputDim = actInfo.Dimension(1);
                        
            layers = [
                sequenceInputLayer(inputDim, 'Name', 'input')  
                
                selfAttentionLayer(8, 256, 'Name', 'enc_self_attn1') 
                additionLayer(2, 'Name', 'enc_add1')                  
                layerNormalizationLayer('Name', 'enc_ln1')            
                fullyConnectedLayer(256*2, 'Name', 'enc_fc1')        
                reluLayer('Name', 'enc_relu1')                        
                fullyConnectedLayer(inputDim, 'Name', 'enc_fc2')         
                additionLayer(2, 'Name', 'enc_add2')                  
                layerNormalizationLayer('Name', 'enc_ln2')            

                selfAttentionLayer(8, 256, 'Name', 'enc_self_attn2')  
                additionLayer(2, 'Name', 'enc_add3')
                layerNormalizationLayer('Name', 'enc_ln3')
                fullyConnectedLayer(256*2, 'Name', 'enc_fc3')
                reluLayer('Name', 'enc_relu2')
                fullyConnectedLayer(inputDim, 'Name', 'enc_fc4')       
                additionLayer(2, 'Name', 'enc_add4')
                layerNormalizationLayer('Name', 'enc_ln4')
                
                selfAttentionLayer(8, 256, 'AttentionMask', 'causal', 'Name', 'dec_self_attn')
                additionLayer(2, 'Name', 'dec_add1')
                layerNormalizationLayer('Name', 'dec_ln1')
                
                attentionLayer(4, 'Name', 'dec_cross_attn')  
                layerNormalizationLayer('Name', 'dec_ln2')
                
                fullyConnectedLayer(256*2, 'Name', 'dec_fc1')
                reluLayer('Name', 'dec_relu1')
                fullyConnectedLayer(inputDim, 'Name', 'dec_fc2')      
                additionLayer(2, 'Name', 'dec_add3')
                layerNormalizationLayer('Name', 'dec_ln3')
               
                fullyConnectedLayer(256, 'Name', 'output_fc1')   
                reluLayer('Name', 'output_relu')
                fullyConnectedLayer(outputDim, 'Name', 'final_fc')
                sigmoidLayer('Name', 'prob_output')                 
            ];
            lgraph = layerGraph(layers);

            lgraph = connectLayers(lgraph, 'input', 'enc_add1/in2');   % 跳跃连接输入

            lgraph = connectLayers(lgraph, 'enc_ln1', 'enc_add2/in2');

            lgraph = connectLayers(lgraph, 'enc_ln2', 'enc_add3/in2');
            lgraph = connectLayers(lgraph, 'enc_ln3', 'enc_add4/in2');
            lgraph = connectLayers(lgraph, 'input', 'dec_add1/in2');  % 假设共享输入维度
            lgraph = connectLayers(lgraph, 'enc_ln4', 'dec_cross_attn/key');
            lgraph = connectLayers(lgraph, 'enc_ln4', 'dec_cross_attn/value');
            lgraph = connectLayers(lgraph, 'dec_ln2', 'dec_add3/in2');
            
            net = dlnetwork(lgraph);
        end

        function criticNet = createCriticNetwork(obj, obsInfo, actInfo)
            % Define observation and action paths
            obsPath = sequenceInputLayer(obsInfo.Dimension(1), 'Name', 'obsInput');
            actPath = sequenceInputLayer(actInfo.Dimension(1), 'Name', 'actionInput');
            
            % Define common path: concatenate along first dimension.
            commonPath = [
                concatenationLayer(1,2,Name="concat")
                fullyConnectedLayer(50)
                reluLayer
                fullyConnectedLayer(1)
                ];
            
            criticNet = dlnetwork;
            criticNet = addLayers(criticNet, obsPath);
            criticNet = addLayers(criticNet, actPath);
            criticNet = addLayers(criticNet, commonPath);
            
            criticNet = connectLayers(criticNet,"obsInput","concat/in1");
            criticNet = connectLayers(criticNet,"actionInput","concat/in2");
            criticNet = initialize(criticNet);
        end
        
        function action = sampleAction(obj, actionProbs, epsilon)
            % 带epsilon探索的动作采样
            if rand < epsilon
                % 随机探索
                action = rand(size(actionProbs));
            else
                % 按概率采样
                sampled = rand(size(actionProbs)) < actionProbs;
                action = double(gather(extractdata(sampled)));
            end
        end

        
        function [states, actions, rewards, nextStates, dones] = unpackExperience(obj, miniBatch)
            stateCells = miniBatch.Observations;
            stateArray = cat(3, stateCells{:});
            states = dlarray(stateArray, 'CTB');
        
            actionCells = miniBatch.Actions; 
            actionArray = cellfun(@squeeze, actionCells, 'UniformOutput', false);
            actionArray = cat(3, actionArray{:});
            actions = dlarray(actionArray, 'CTB'); 
        
            rewards = dlarray(miniBatch.Rewards', 'CB'); 
        
            nextstateCells = miniBatch.NextObservations;
            nextstateArray = cat(3, nextstateCells{:});
            nextStates = dlarray(nextstateArray, 'CTB');
        
            dones = dlarray(double(miniBatch.IsDone)', 'CB'); 
        end
    end
end

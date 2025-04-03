classdef TemplateMetaOptimizer < rl.agent.CustomAgent
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        PolicyNetwork % Critic
        OptimizerOptions
        ExperienceBuffer = struct(...
            'Observations', {}, ...
            'Actions', {}, ...
            'Rewards', {}, ...
            'NextObservations', {}, ...
            'IsDone', {})
    end
    %% required
    methods
        function obj = TemplateMetaOptimizer(observationInfo, actionInfo)
            obj = obj@rl.agent.CustomAgent(observationInfo, actionInfo);

            validateattributes(observationInfo, {'rl.util.RLDataSpec'}, {'scalar'}, '', 'observationInfo');
            validateattributes(actionInfo, {'rl.util.RLDataSpec'}, {'scalar'}, '', 'actionInfo');
            
            
            obj.PolicyNetwork = createPolicyNetwork(obj, observationInfo, actionInfo); 
            obj.OptimizerOptions = struct(...
                'LearnRate', 1e-3, ...
                'GradientThreshold', 1, ...
                'BatchSize', 32);
        end
    end
    %% required
    methods (Access = protected)
        function action = getActionImpl(obj, observation)
            
            obsDL = dlarray(observation, 'CB');  
            actionDL = forward(obj.PolicyNetwork, obsDL);
           
            action = extractdata(actionDL);
            
            % 添加探索噪声（示例使用高斯噪声）
            explorationNoise = 0.1 * randn(size(action));
            action = action + explorationNoise;
            
            % 确保动作在有效范围内
            action = max(min(action, obj.ActionInfo.UpperLimit), ...
                        obj.ActionInfo.LowerLimit);
        end
        
        function learnImpl(obj, experience)
            obj.ExperienceBuffer(end+1) = experience;
            if numel(obj.ExperienceBuffer) >= obj.OptimizerOptions.BatchSize
                batchIdx = randperm(numel(obj.ExperienceBuffer), obj.OptimizerOptions.BatchSize);
                batchData = obj.ExperienceBuffer(batchIdx);
                
                obsBatch = dlarray(cat(2, batchData.Observation), 'CB');
                actionBatch = dlarray(cat(2, batchData.Action), 'CB');
                rewardBatch = dlarray(cat(2, batchData.Reward), 'CB');
                
                [loss, gradients] = dlfeval(@obj.computeLoss, ...
                                          obsBatch, actionBatch, rewardBatch);
                
                obj.PolicyNetwork = dlupdate(...
                    @(w,g) w - obj.OptimizerOptions.LearnRate * g, ...
                    obj.PolicyNetwork, ...
                    gradients);
                
                obj.ExperienceBuffer = [];
            end
        end
        
        function resetImpl(obj)
            obj.ExperienceBuffer = [];
            if isa(obj.PolicyNetwork, 'nnet.cnn.LayerGraph')
                obj.PolicyNetwork = resetState(obj.PolicyNetwork);
            end
        end
    end
    %% chosen
    methods (Access = private)
        function net = createPolicyNetwork(obj, obsInfo, actInfo)
            inputSize = prod(obsInfo.Dimension);
            outputSize = prod(actInfo.Dimension);
            
            layers = [
                featureInputLayer(inputSize, 'Name', 'input')
                fullyConnectedLayer(64, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(64, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(outputSize, 'Name', 'output')
            ];
            
            net = dlnetwork(layers);
        end
        
        function [loss, gradients] = computeLoss(obj, obs, action, reward)
            predictedAction = forward(obj.PolicyNetwork, obs);
            advantage = reward; 
            logProb = -0.5 * sum((predictedAction - action).^2);  
            loss = -mean(logProb .* advantage);
            
            gradients = dlgradient(loss, obj.PolicyNetwork.Learnables);
            
            gradients = dlupdate(@(g) ...
                g ./ max(1, norm(g)/obj.OptimizerOptions.GradientThreshold), ...
                gradients);
        end
    end
    % chosen
    methods (Access = protected)
        function validateEnvironmentImpl(~)
           
        end
        
        function setupImpl(obj)
            
        end
        
        function releaseImpl(obj)
           
        end
    end
end

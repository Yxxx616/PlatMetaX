classdef MLP_Alg_Rec_Metaoptimizer < rl.agent.CustomAgent
    % MetaOptimizerRL 基于强化学习的元优化器模板类
    % 继承自rl.agent.CustomAgent，实现必要接口
    
    properties
        mlpNN
        curaction = 1
        expBuffer
        sampleBuffer
        expCount
        sampleCount
        layers
        options
    end
    
    methods
        function obj = MLP_Alg_Rec_Metaoptimizer(observationInfo, actionInfo)
            obj = obj@rl.agent.CustomAgent();
            obj.ObservationInfo = observationInfo;
            obj.ActionInfo = actionInfo;  
            obj.expCount = 1;
            obj.sampleCount = 1;
            hiddenLayerSize = 64;  % 隐藏层大小
            obj.mlpNN = fitnet(hiddenLayerSize);
        end
    end
    methods (Access = protected)
        function action = getActionImpl(obj,observation)
            preaction = classify(obj.mlpNN,observation{1});
            action = double(preaction);
        end
        function action = getActionWithExplorationImpl(obj, observation)
            if obj.curaction > max(obj.ActionInfo.Elements)
                obj.curaction = 1;
            end 
            action = obj.curaction;
            obj.curaction = obj.curaction + 1;
        end
        function action = learnImpl(obj, experience)
            proFeature = experience{1}{1};
            algidx = experience{2}{1};
            algScore = experience{3};  
            obj.expBuffer(obj.expCount).proFeature = proFeature;
            obj.expBuffer(obj.expCount).algidx = algidx;
            obj.expBuffer(obj.expCount).score = algScore;
            
            if algidx >= 5
                tempscore = [obj.expBuffer(obj.expCount-5+1:obj.expCount).score];
                [~,label] = max(tempscore);
                obj.sampleBuffer(obj.sampleCount).xFeature = obj.expBuffer(obj.expCount).proFeature;
                obj.sampleBuffer(obj.sampleCount).yLabel = label;
                obj.sampleCount = obj.sampleCount + 1;
            end
            
            obj.expCount = obj.expCount + 1; 
            if obj.sampleCount > 70
                X = cat(1,obj.sampleBuffer.xFeature);  %for training, X need to be (numSamples * numFeatures)
                Y = [obj.sampleBuffer.yLabel];
                Y = categorical(Y);
                categoriesY = categories(Y);
                numCategories = numel(categoriesY);
                
                %split dataSet
                numObs = size(X,1);
                numTrain = floor(numObs*0.85);
                numTest = numObs-numTrain;
                
                idx = randperm(numObs);
                idxTrain = idx(1:numTrain);
                idxTest = idx(numTrain+1:end);
                
                XTrain = X(idxTrain,:);
                XTest = X(idxTest,:);
                
                YTrain = Y(idxTrain);
                YTest = Y(idxTest);
                
                obj.layers = [
                    featureInputLayer(obj.ObservationInfo.Dimension(1),'Normalization', 'zscore')   % Input layer
                    fullyConnectedLayer(64, 'Name', 'fc1')         % Fully connected layer with 64 neurons
                    reluLayer('Name', 'relu1')                    % ReLU activation function
                    fullyConnectedLayer(32, 'Name', 'fc2')         % Fully connected layer with 32 neurons
                    reluLayer('Name', 'relu2')                    % ReLU activation function
                    fullyConnectedLayer(numCategories, 'Name', 'fc3') % Output layer
                    softmaxLayer('Name', 'softmax')                % Softmax layer for classification
                    classificationLayer('Name', 'output')          % Classification output layer
                ];
                
                obj.options = trainingOptions('sgdm', ...
                    'MaxEpochs', 100, ...
                    'MiniBatchSize', 10, ...
                    'InitialLearnRate', 0.01, ...
                    'Shuffle', 'every-epoch', ...
                    'Verbose', false, ...
                    'Plots', 'training-progress', ...
                    'ValidationData', {XTest, YTest}, ...
                    'ValidationFrequency', 30, ...
                    'ValidationPatience', 10);
                
                % 获取当前脚本的路径
                currentScriptPath = mfilename('fullpath');
                % 获取当前脚本的上上级目录路径
                parentDir = fileparts(fileparts(fileparts(currentScriptPath)));
                % 拼接目标文件夹路径
                targetFolder = fullfile(parentDir, 'TSP_Instances');
                saveFilePath = fullfile(targetFolder, 'data.mat');
                save(saveFilePath, 'X', 'Y');
                obj.mlpNN = trainNetwork(XTrain, YTrain, obj.layers,obj.options);
            end
            action = getActionWithExplorationImpl(obj, experience{4});
        end  
        
        function resetImpl(obj)
            % 重置智能体状态
            % 初始化或重置内部状态变量
            obj.expCount = 1;
            obj.sampleCount = 1;
        end
    end
end

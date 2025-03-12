classdef Train < handle
    properties
        MetaOptimizer
        BaseOptimizer
        TrainingSet
        TrainingSetName
        epoch = 1
        env
    end
    methods
        function obj = Train(mo, bo, envName, problemset)
            obj.BaseOptimizer = bo();
            obj.TrainingSetName = problemset.psName;
            [obj.TrainingSet, ~] = splitProblemSet(problemset);
            obj.env = envName(obj.TrainingSet,obj.BaseOptimizer,'train');
            obsInfo = getObservationInfo(obj.env);
            actInfo = getActionInfo(obj.env);
            obj.MetaOptimizer = mo(obsInfo,actInfo);
        end
        
        function result = run(obj)
            set(0, 'DefaultFigureVisible', 'off');
            time_str = datestr(datetime('now'), 'yyyymmddHHMMSS');
            folderName = [class(obj.MetaOptimizer), '_', ...
              obj.TrainingSetName, '_D', ...
              num2str(obj.TrainingSet{1}.D), '_', ...
              time_str];
            currentScriptPath = mfilename('fullpath');
            % 获取当前脚本所在的目录
            currentDir = fileparts(currentScriptPath);
            % 构建 SaveAgentDirectory 的完整路径
            saveLogDir = fullfile(currentDir, 'Data', 'TrainingLog',folderName);

            % 确保目录存在
            if ~exist(saveLogDir, 'dir')
                mkdir(saveLogDir);
            end
            
            trainOpts = rlTrainingOptions(...
                'MaxEpisodes',1000,...
                'MaxStepsPerEpisode',1000,...
                'SaveAgentCriteria','EpisodeReward',...
                'SaveAgentValue',0,...
                'ScoreAveragingWindowLength',10,...
                'SaveAgentDirectory',saveLogDir,...
                'Plots', 'none'); %close 
            trainingInfo = train(obj.MetaOptimizer,obj.env,trainOpts);
            agent = obj.MetaOptimizer;
            save( [saveLogDir, '\',class(obj.MetaOptimizer),'_finalAgent.mat'],'agent');
            save( [currentDir,'\AgentModel\', class(obj.MetaOptimizer),'_finalAgent.mat'],'agent');
            result.trainingInfo = trainingInfo;
        end
    end
end

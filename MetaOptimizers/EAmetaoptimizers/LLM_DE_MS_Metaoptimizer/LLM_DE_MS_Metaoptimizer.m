classdef LLM_DE_MS_Metaoptimizer < rl.agent.CustomAgent
% Algorithm generation-mutation strategy generation
% python-LLM as meta-optimizer, being trained via EL.
% DE as base-optimizer, with mutation strategy generation as opotimization object.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: A MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        metaTable
        metaObj
        indcount
        metapop
        metaNP = 5
        llm
        strategyCount
        temprecord
        history
        proNum = 18
    end
    
    methods
        function obj = LLM_DE_MS_Metaoptimizer(observationInfo, actionInfo)
            obj = obj@rl.agent.CustomAgent();
%             addpath('E:\yx\toolbox\llmMatlab\llms-with-matlab')
            currentFilePath = mfilename('fullpath');
            [currentFolder, ~, ~] = fileparts(currentFilePath);
            if count(py.sys.path, currentFolder) == 0
                insert(py.sys.path, int32(0), currentFolder);
            end
            mod = py.importlib.import_module('testdeepseek');
            py.importlib.reload(mod);
            obj.ObservationInfo = observationInfo;
            obj.ActionInfo = actionInfo;
            obj.history = {};
            for i = 1:obj.proNum %store best population of $28$ probloms 
                obj.metaTable(i).population = []; 
                obj.metaTable(i).fitness = -Inf * ones(obj.metaNP,1);
                obj.metapop(i).population = [];
                obj.metapop(i).fitness = []; 
            end
            obj.strategyCount = 1;
            obj.indcount = 1;
            
        end
    end
    methods (Access = protected)
        function action = getActionImpl(obj,observation)
            obs = observation{1};
            if obs(end) <= obj.proNum
                % 如果键存在，获取对应的值
                curpop = obj.metaTable(obs(end)).population;
                curobj = obj.metaTable(obs(end)).fitness;
                [~,maxidx]=max(curobj);
                action = curpop(maxidx,:);
            else
                action = randperm(obj.strategyCount-1);
            end
        end
        function action = getActionWithExplorationImpl(obj, observation)
            if obj.indcount>obj.metaNP
                obj.indcount = 1;
            end
            tempFile = [pwd '\BaseOptimizers\Single-objective optimization\Learned\LLM_DE_MS_Baseoptimizer\updateFunc', num2str(obj.strategyCount), '.m'];
            
%             obs = observation{1};
%             popdecs = reshape(obs(1:500),50,10);
%             popfits = reshape(obs(501:550),50,1);
%             popcv = reshape(obs(551:600),50,1);
%             response = py.testdeepseek.process_string(mat2str(popdecs),mat2str(popfits),mat2str(popcv),num2str(obs(end)),num2str(obj.strategyCount),obj.loadHistory(obs(end),obj.strategyCount));
%             matlabCode = char(response);
%             obj.temprecord.code = matlabCode;
%             fid = fopen(tempFile, 'w');
%             fprintf(fid, '%s', matlabCode);
%             fclose(fid);

            if ~exist(tempFile, 'file') 
                obs = observation{1};
                popdecs = reshape(obs(1:500),50,10);
                popfits = reshape(obs(501:550),50,1);
                popcv = reshape(obs(551:600),50,1);
                response = py.testdeepseek.process_string(mat2str(popdecs),mat2str(popfits),mat2str(popcv),num2str(obs(end)),num2str(obj.strategyCount),obj.loadHistory(obs(end),obj.strategyCount));
                matlabCode = char(response);
                obj.temprecord.code = matlabCode;
                fid = fopen(tempFile, 'w');
                fprintf(fid, '%s', matlabCode);
                fclose(fid);
            else
                obj.temprecord.code = fileread(tempFile);
            end
            action = obj.strategyCount;
            obj.indcount = obj.indcount + 1;
            obj.strategyCount = obj.strategyCount + 1;
            
        end
        function action = learnImpl(obj, experience)
            obs = experience{1}{1};
            pid = obs(end);
            ind = experience{2}{1};
            fitness = experience{3};  
            obj.metapop(pid).population = [obj.metapop(pid).population; ind];
            obj.metapop(pid).fitness = [obj.metapop(pid).fitness;fitness];
            obj.temprecord.perf = fitness;
            obj.history{obj.strategyCount-1}=obj.temprecord;
            if experience{5}
                try
                    for i = 1:obj.proNum
                        % 找到需要替换的索引
                        replaceIdx = find(obj.metaTable(i).fitness < obj.metapop(i).fitness);

                        % 替换 population 和 fitness
                        obj.metaTable(i).population(replaceIdx, :) = obj.metapop(i).population(replaceIdx, :);
                        obj.metaTable(i).fitness(replaceIdx) = obj.metapop(i).fitness(replaceIdx);

                        % 清空 metapop 的 population 和 fitness
                        obj.metapop(i).population = [];
                        obj.metapop(i).fitness = [];
                    end
                catch ME
                    % 捕获错误并输出调试信息
                    disp('发生错误：');
                    disp(ME.message); % 输出错误信息
                    disp('错误发生在以下位置：');
                    disp(ME.stack(1)); % 输出错误位置
                end
            end
            action = getActionWithExplorationImpl(obj,experience{4});
        end  
        function history = loadHistory(obj,functionnum,current_iter)
            offset = (functionnum - 1) * obj.metaNP;
            base_step = obj.proNum * obj.metaNP;

            if mod((current_iter - offset),base_step) == 1 || isempty(obj.history) % && (0 <= floor((current_iter - offset) / base_step)) && (floor((current_iter - offset) / base_step)< num_bases) ||  isempty(obj.history)
                history = "No history";
            else
                last = obj.history{end};
                history = strcat('Your generated last code for this problem:' , last.code, newline, 'This mutation strategy made the algorithm code:',num2str(last.perf),'(the bigger, the better)');
            end
        end
        function resetImpl(obj)
            
        end
    end
end

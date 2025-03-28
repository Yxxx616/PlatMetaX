classdef DE_DE_FCR_Metaoptimizer < rl.agent.CustomAgent
    % MetaOptimizerRL 基于强化学习的元优化器模板类
    % 继承自rl.agent.CustomAgent，实现必要接口
    
    properties
        metaTable
        metaObj
        indcount
        metapop
        metaNP = 10
    end
    
    methods
        function obj = DE_DE_FCR_Metaoptimizer(observationInfo, actionInfo)
            obj = obj@rl.agent.CustomAgent();
            for i = 1:observationInfo.UpperLimit
                obj.metaTable(i).population= rand(obj.metaNP,actionInfo.Dimension(1));
                obj.metaTable(i).fitness = -Inf * ones(obj.metaNP,1);
                obj.metapop(i).population = [];
                obj.metapop(i).fitness = [];
            end
            obj.ObservationInfo = observationInfo;
            obj.ActionInfo = actionInfo;
            obj.indcount = 1;
        end
    end
    methods (Access = protected)
        function action = getActionImpl(obj,observation)
            if observation{1} <= obj.ObservationInfo.UpperLimit
                % 如果键存在，获取对应的值
                curpop = obj.metaTable(observation{1}).population;
                curobj = obj.metaTable(observation{1}).fitness;
                [~,maxidx]=max(curobj);
                action = curpop(maxidx,:);
            else
                action = rand(obj.ActionInfo.Dimension);
            end
        end
        function action = getActionWithExplorationImpl(obj, observation)
            if obj.indcount>obj.metaNP
                obj.indcount = 1;
            end
            if observation{1} <= obj.ObservationInfo.UpperLimit
                curpop = obj.metaTable(observation{1}).population;
                p1 = curpop(randperm(obj.metaNP,1),:);
                p2 = curpop(randperm(obj.metaNP,1),:);
                action = curpop(obj.indcount,:);
            else
                p1 = rand(obj.ActionInfo.Dimension);
                p2 = rand(obj.ActionInfo.Dimension);
                action = rand(obj.ActionInfo.Dimension);
            end

            Site = rand(size(action)) < 0.8;
            Offspring = action;
            Offspring(Site) = Offspring(Site) + 0.5*(p1(Site)-p2(Site));
            action = Offspring;
   
            obj.indcount = obj.indcount + 1;
        end
        function action = learnImpl(obj, experience)
            pid = experience{1}{1};
            ind = experience{2}{1};
            fitness = experience{3};  
            obj.metapop(pid).population = [obj.metapop(pid).population; ind];
            obj.metapop(pid).fitness = [obj.metapop(pid).fitness;fitness]; 
            if experience{5}
                try
                    for i = 1:obj.ObservationInfo.UpperLimit
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
        
        function resetImpl(obj)
            % 重置智能体状态
            % 初始化或重置内部状态变量
            for i = 1:obj.ObservationInfo.UpperLimit
                obj.metaTable(i).population= rand(obj.metaNP,obj.ActionInfo.Dimension(1));
                obj.metaTable(i).fitness = -Inf * ones(obj.metaNP,1);
                obj.metapop(i).population = [];
                obj.metapop(i).fitness = [];
            end
            obj.indcount = 1;
        end
    end
end

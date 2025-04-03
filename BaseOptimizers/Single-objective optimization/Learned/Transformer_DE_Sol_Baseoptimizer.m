classdef Transformer_DE_Sol_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none> <learned/none>

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: A MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        Population      % 当前种群
        BestSolution    % 历史最佳解
        CR              % 交叉概率
        F               % 缩放因子
        StrategyParams  % 变异策略参数
        Problem         % 当前优化问题
        
        prevFeasible
        prevBest
        RewardWeights = [0.7, 0.3] % 可行率与最优值权重
    end
    
    methods        
        function init(this, problem)
            % 初始化优化器
            this.Problem = problem;
            this.Population = problem.Initialization();
            [this.BestSolution, ~] = this.findBestIndividual();
            this.prevFeasible =  this.calculateFeasibleRate();
            this.prevBest = this.BestSolution.obj;
        end
        
        function [reward, done] = update(this, action)
            offspring = action(:,:)';
            Offspring = this.pro.Evaluation(offspring);
            replace             = FitnessSingle(this.Population) > FitnessSingle(Offspring);
            this.Population(replace) = Offspring(replace);

            done = ~this.NotTerminated(this.Population);
            [newBest, ~] = this.findBestIndividual();
            
            % 计算奖励
            newFeasible = this.calculateFeasibleRate();
            deltaFeasible = newFeasible - this.prevFeasible;
            deltaBest = (this.prevBest - newBest.obj)/abs(this.prevBest);

            this.prevFeasible = newFeasible;
            this.prevBest = newBest.obj;
            
            reward = this.RewardWeights(1)*deltaFeasible + ...
                     this.RewardWeights(2)*deltaBest;
        end
        
        function [best, idx] = findBestIndividual(this)
            % 找到当前最优解
            [~, idx] = min(this.Population.objs);
            best = this.Population(idx);
        end
    end

    methods (Access = private)
        function feasibleRate = calculateFeasibleRate(this)
            % 计算可行率
            cvs = sum(max(0,this.Population.cons),2);
            feasibleRate = sum(cvs <= 0) / numel(cvs);
        end
        
    end
end


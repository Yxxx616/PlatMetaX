classdef DQN_DE_AS_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none> <learned/none>
% USE DQN TO SELECT Differential Evolution series.

%------------------------------- Reference --------------------------------
% R. Storn and K. Price, Differential evolution-a simple and efficient
% heuristic for global optimization over continuous spaces, Journal of
% Global Optimization, 1997, 11(4): 341-359.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        consNum
        Population
        NP
        baseperformance
        curBP
        bestInd
        
        CR
        F
    end
    methods
        function init(this,problem)
            %% Parameter setting
            [this.CR,this.F] = this.ParameterSet(0.9,0.5);
            
            %% Generate random population
            this.Population = problem.Initialization();
            
            [~,this.consNum] = size(this.Population.cons);
            this.NP = problem.N;
            [this.baseperformance,idx] = min(this.Population.objs);
            this.curBP = this.baseperformance;
            this.bestInd = this.Population(idx);
        end
        
        function [reward, nextState, done, bestPop] = update(this,BOparameters,Problem)
            if BOparameters <= 4
                while this.NotTerminated(this.Population)
                    MatingPool = TournamentSelection(2,2*Problem.N,this.Population.objs);
                    Offspring  = LearnedMSOperatorDE(Problem,this.Population,this.Population(MatingPool(1:end/2)),this.Population(MatingPool(end/2+1:end)),this.bestInd, BOparameters,{this.CR,this.F,0,0});
                    replace             = FitnessSingle(this.Population) > FitnessSingle(Offspring);
                    this.Population(replace) = Offspring(replace);
                    finalPop = this.Population;
                    
                end
            else
                set(0, 'DefaultFigureVisible', 'off');
                if BOparameters == 5
                    alg = IMODE();
                elseif BOparameters == 6
                    alg = SHADE();
                end
                alg.Solve(this.pro);
                finalPop = alg.result{end,2};
            end
            try
                reward = (this.baseperformance - min(finalPop.objs))/abs(this.baseperformance);
            catch
                reward = -Inf;
            end
            done = true;
            nextState = 0;
            bestPop = this.Population;   
        end

        function state = calState(this)
            objs = this.Population.objs;
            cons = this.Population.cons;
            mean_obj = mean(objs); 
            std_obj = std(objs);
            best_obj = min(objs);
            worst_obj = max(objs); 
            
            [N, p] = size(cons);
            cv = max(cons, 0); % 计算每个个体的约束违反度
            mean_cv = mean(sum(cv, 2)); % 平均约束违反度
            feasible_ratio = sum(all(cons <= 0, 2)) / N; % 约束满足个体比例
            
            cons_pattern = double(cons > 0); % 约束违反模式的二进制向量
            cons_diversity = 0;
            for i = 1:N
                for j = i+1:N
                    cons_diversity = cons_diversity + sum(abs(cons_pattern(i, :) - cons_pattern(j, :)));
                end
            end
            cons_diversity = cons_diversity / (N * (N - 1) / 2); % 汉明距离的平均值
            
            cons_adaptability = zeros(N, 1);
            for i = 1:N
                feasible_cons = sum(cons(i, :) <= 0); % 个体i满足的约束数量
                cons_adaptability(i) = objs(i) * feasible_cons / p; % 约束适应性指数
            end
            cons_adaptability = mean(cons_adaptability); % 种群的约束适应性指数
            
            feasible_objs = objs(all(cons <= 0, 2));
            infeasible_objs = objs(any(cons > 0, 2));

            % 计算均值差异
            mean_feasible = mean(feasible_objs);
            mean_infeasible = mean(infeasible_objs);
            delta_mean = abs(mean_feasible - mean_infeasible);

            % 计算标准差差异
            std_feasible = std(feasible_objs);
            std_infeasible = std(infeasible_objs);
            delta_std = abs(std_feasible - std_infeasible);

            % 计算分布均匀性差异（使用Delta'指标）
            [delta_feasible, ~] = calculate_delta_prime(feasible_objs);
            [delta_infeasible, ~] = calculate_delta_prime(infeasible_objs);
            delta_dist = abs(delta_feasible - delta_infeasible);

            
            % 计算卡方检验指标
            q = 10; % 将目标值空间划分为q+1个区域
            [counts_feasible, edges] = histcounts(feasible_objs, q+1);
            [counts_infeasible, ~] = histcounts(infeasible_objs, edges);
            sigma = counts_feasible + counts_infeasible;
            chi2 = sum((counts_feasible - counts_infeasible).^2 ./ sigma);

            state = [mean_obj,std_obj,best_obj,worst_obj,...
                mean_cv,feasible_ratio,cons_diversity,cons_adaptability,...
                delta_mean, delta_std, delta_dist, chi2];
        end
    end
end


function [delta_prime, mean_d] = calculate_delta_prime(objs)
    sorted_objs = sort(objs);
    N = length(sorted_objs);
    d = diff(sorted_objs);
    mean_d = mean(d);
    delta_prime = sum(abs(d - mean_d)) / (N - 1);
end
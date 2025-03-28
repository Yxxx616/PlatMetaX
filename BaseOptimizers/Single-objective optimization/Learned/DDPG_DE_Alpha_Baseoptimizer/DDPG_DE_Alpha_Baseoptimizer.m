classdef DDPG_DE_Alpha_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none> <learned/none>
% USE DDPG TO SET THE Alpha value for CHT
% CR --- 0.9 --- Crossover constant
% F  --- 0.5 --- Mutation factor

%------------------------------- Reference --------------------------------
% R. Storn and K. Price, Differential evolution-a simple and efficient
% heuristic for global optimization over continuous spaces, Journal of
% Global Optimization, 1997, 11(4): 341-359.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025.
%--------------------------------------------------------------------------
    properties
        consNum
        Population
        NP
        baseperformance
        curBP
        F
        CR
    end
    methods
        function init(this,problem)
            %% Parameter setting
            this.CR = 0.9;
            this.F = 0.5;
            %% Generate random population
            this.Population = problem.Initialization();
            
            [~,this.consNum] = size(this.Population.cons);
            this.NP = problem.N;
            this.baseperformance = min(this.Population.objs);
            this.curBP = this.baseperformance;
        end
        
        function [reward, nextState, done, bestPop] = update(this,BOparameters,Problem)
            MatingPool = TournamentSelection(2,2*Problem.N,FitnessSingle(this.Population));
            Offspring  = OperatorDE(Problem,this.Population,this.Population(MatingPool(1:end/2)),this.Population(MatingPool(end/2+1:end)),{this.CR,this.F,0,0});
            replace             = FitnessWithAdaptiveAlphaCHT(this.Population,BOparameters) > FitnessWithAdaptiveAlphaCHT(Offspring,BOparameters);
            this.Population(replace) = Offspring(replace);
            nextState = this.calState();
            nofinish = this.NotTerminated(this.Population);
            done = ~nofinish;
            currentBP = min(this.Population.objs);
            if isnan(currentBP)
                currentBP = this.curBP;
                reward = -10;
            elseif currentBP - this.curBP > 0
                reward = -1;
            elseif currentBP - this.curBP == 0
                reward  = 0;
            else
                reward = 1;
            end
            this.curBP = currentBP;
            this.baseperformance = min(currentBP,this.baseperformance);
            if this.baseperformance > 1e-8
                reward = reward - 1;
            end
            if done
                bestPop = this.Population;
                if this.baseperformance < 1e-8
                    reward = reward + 100;
                end
            else
                bestPop = 0;
            end
        end
        
        function state = calState(this)
        %计算约束单目标优化特征
            % 识别可行点
            state = zeros(14,1);
            consSum = sum(this.Population.cons,2);
            objVec = this.Population.objs;
            feasible_mask = any(this.Population.cons<=0,2);
            if sum(feasible_mask)> 0
                feasible_points = this.Population(feasible_mask).decs;
                % 使用DBSCAN聚类算法识别可行组件
                epsilon = 0.5; % 邻域半径，需要根据数据调整
                minpts = 10; % 邻域内最小点数，需要根据数据调整
                [idx, ~] = dbscan(feasible_points, epsilon, minpts);

                % 计算可行组件数量
                NF = numel(unique(idx)) - 1; % 减1是因为0是为噪声点分配的
                state(1) = NF/ sum(feasible_mask);
                % 计算可行性比率
                qF = sum(feasible_mask) / this.NP;
                state(2) = qF;

                % 计算可行边界交叉比率
                % 计算边界交叉数量
                crossings = sum(diff([0, feasible_mask', 0]) ~= 0);
                RFBx = crossings / (this.NP - 1);
                state(3) = RFBx;
            end

            FVC = corr(objVec, consSum, 'Type', 'Spearman');
            state(4) = FVC;
            minFitness = min(objVec, [], 1);
            maxFitness = max(objVec, [], 1);

            % 计算约束违反程度的最小值和最大值
            minViolation = min(consSum);
            maxViolation = max(consSum);
            state(5) = minFitness / (maxFitness+1e-6);
            state(6) = minViolation / (maxViolation+1e-6);

            % 确定理想区域的范围
            % 对于每个目标，理想区域的范围是目标最小值加上25%的目标范围
            % 对于约束违反程度，理想区域的范围是约束最小值加上25%的约束范围
            idealZoneFitness = minFitness + 0.25 * (maxFitness - minFitness);
            idealZoneViolation = minViolation + 0.25 * (maxViolation - minViolation);

            % 计算在理想区域内的点的数量
            inIdealZone = sum(all(objVec <= idealZoneFitness, 2) & consSum <= idealZoneViolation);

            % 计算理想区域比例
            PiIZ0_25 = inIdealZone / this.NP;

            % 对于 1% 的理想区域，可以类似地计算
            idealZoneFitness = minFitness + 0.01 * (maxFitness - minFitness);
            idealZoneViolation = minViolation + 0.01 * (maxViolation - minViolation);
            inIdealZone = sum(all(objVec <= idealZoneFitness, 2) & consSum <= idealZoneViolation);
            PiIZ0_01 = inIdealZone / this.NP;

            state(7) = PiIZ0_25;
            state(8) = PiIZ0_01;
            state(9) = sum(consSum);
            state(10) = minFitness;
            state(11) = maxFitness;
            state(12) = minViolation;
            state(13) = maxViolation;
            state(14) = mean(consSum);
        end
    end
end
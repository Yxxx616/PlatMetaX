classdef DDPG_NSGAII_Alpha_Baseoptimizer < BASEOPTIMIZER
% <2025> <learned/none> <multi> <real/integer/label/binary/permutation> <constrained/none>
% Nondominated sorting genetic algorithm II
    properties
        curProblem
        consNum
        Population
        NP
        hvValue
        
        rankNo
        FrontNo
        CrowdDis
    end
    methods
        function init(this,problem)
            this.curProblem = problem;
            this.Population = problem.Initialization();
            [~,this.consNum] = size(this.Population.cons);
            this.NP = problem.N;
            this.hvValue = 0;
            [~,this.FrontNo,this.CrowdDis] = DNAEnvironmentalSelection(this.Population,problem.N,0);
        end
        
        function [reward, nextState, done, bestPop] = update(this,BOparameters,Problem)
            MatingPool = TournamentSelection(2,Problem.N,this.FrontNo,-this.CrowdDis);
            Offspring  = OperatorGA(Problem,this.Population(MatingPool));
            [this.Population,this.FrontNo,this.CrowdDis] = DNAEnvironmentalSelection([this.Population,Offspring],Problem.N,BOparameters);
            
            nextState = calMOPState(this);
            nofinish = this.NotTerminated(this.Population);
            done = ~nofinish;
            curHV = Problem.CalMetric('HV',this.Population);
            if isnan(curHV)
                curHV = 0;
                reward = -1;
            elseif curHV - this.hvValue <=0
                reward = 0;
            else
                reward = 1;
            end
            this.hvValue = curHV;
            if done
                bestPop = this.Population;
            else
                bestPop = 0;
            end
        end
        
        function state = calMOPstate(this)
            state = zeros(8,1);
            objVec = this.Population.objs;
            consSum = sum(this.Population.cons,2);
            feasible_mask = any(obj.Population.cons<=0,2);
            if sum(feasible_mask)> 0
                feasible_points = obj.Population(feasible_mask).decs;
                % 使用DBSCAN聚类算法识别可行组件
                epsilon = 0.5; % 邻域半径，需要根据数据调整
                minpts = 10; % 邻域内最小点数，需要根据数据调整
                [idx, ~] = dbscan(feasible_points, epsilon, minpts);

                % 计算可行组件数量
                NF = numel(unique(idx)) - 1; % 减1是因为0是为噪声点分配的
                state(1) = NF/ sum(feasible_mask);
                % 计算可行性比率
                qF = sum(feasible_mask) / obj.NP;
                state(2) = qF;

                % 计算可行边界交叉比率
                % 计算边界交叉数量
                crossings = sum(diff([0, feasible_mask', 0]) ~= 0);
                RFBx = crossings / (obj.NP - 1);
                state(3) = RFBx;
            end

            % 计算信息内容特征
            % 首先，你需要对约束违反值进行排序
            cons_sorted = sort(consSum);
            H = zeros(obj.NP,1);
            probabilities = hist(cons_sorted, unique(cons_sorted)) / obj.NP;
            for i = 1:length(probabilities)
                p = probabilities(i);
                if p > 0
                    H(i) =  - p * log(p);
                end
            end

            % 计算信息内容的最大值（Hmax）
            Hmax = max(H);
            state(4) = Hmax;

            minFitness = min(objVec, [], 1);
            maxFitness = max(objVec, [], 1);

            % 计算约束违反程度的最小值和最大值
            minViolation = min(consSum);
            maxViolation = max(consSum);
            state(5) = mean(consSum);
            state(6) = std(consSum);

            % 确定理想区域的范围
            % 对于每个目标，理想区域的范围是目标最小值加上25%的目标范围
            % 对于约束违反程度，理想区域的范围是约束最小值加上25%的约束范围
            idealZoneFitness = minFitness + 0.25 * (maxFitness - minFitness);
            idealZoneViolation = minViolation + 0.25 * (maxViolation - minViolation);

            % 计算在理想区域内的点的数量
            inIdealZone = sum(all(objVec <= idealZoneFitness, 2) & consSum <= idealZoneViolation);

            % 计算理想区域比例
            PiIZ0_25 = inIdealZone / obj.NP;

            % 对于 1% 的理想区域，可以类似地计算
            idealZoneFitness = minFitness + 0.01 * (maxFitness - minFitness);
            idealZoneViolation = minViolation + 0.01 * (maxViolation - minViolation);
            inIdealZone = sum(all(objVec <= idealZoneFitness, 2) & consSum <= idealZoneViolation);
            PiIZ0_01 = inIdealZone / obj.NP;

            state(7) = PiIZ0_25;
            state(8) = PiIZ0_01;
        end
    end
end
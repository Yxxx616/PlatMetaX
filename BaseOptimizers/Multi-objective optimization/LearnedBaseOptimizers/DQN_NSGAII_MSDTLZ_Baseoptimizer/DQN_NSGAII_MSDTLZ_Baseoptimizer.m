classdef DQN_NSGAII_MSDTLZ_Baseoptimizer < BASEOPTIMIZER
% <2025> <learned> <multi> <real/integer/label/binary/permutation> <constrained/none>
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
        curPF
    end
    methods
        function init(this,problem)
            this.curProblem = problem;
            this.Population = problem.Initialization();
            [~,this.consNum] = size(this.Population.cons);
            this.NP = problem.N;
            this.hvValue = 0;
            
            [~,this.FrontNo,this.CrowdDis] = EnvironmentalSelection(this.Population,problem.N);
            this.curPF = find(this.FrontNo==1);
        end
        
        function [reward, nextState, done, bestPop] = update(this,BOparameters,Problem)
            MatingPool = TournamentSelection(2,2*Problem.N,this.FrontNo,-this.CrowdDis);
            Offspring  = LearnedMSOperatorDEforNSGAII(Problem,this.Population,this.Population(MatingPool(1:end/2)),this.Population(MatingPool(end/2+1:end)),this.Population(this.curPF), BOparameters,{0.9,0.5,0,0});
            [this.Population,this.FrontNo,this.CrowdDis] = EnvironmentalSelection([this.Population,Offspring],Problem.N);
            
            nextState = this.calMOPstate();
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
            fitness = this.Population.objs;
            state = [min(fitness(:,1)),min(fitness(:,2)),max(fitness(:,1)),max(fitness(:,2)),mean(fitness(:,1)),mean(fitness(:,2)),std(fitness(:,1)),std(fitness(:,2))];
        end
    end
end
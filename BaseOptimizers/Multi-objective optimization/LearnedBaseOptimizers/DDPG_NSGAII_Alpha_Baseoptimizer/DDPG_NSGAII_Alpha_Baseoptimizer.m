classdef DDPG_NSGAII_Alpha_Baseoptimizer < BASEOPTIMIZER
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
    end
    methods
        function init(this,problem)
            this.curProblem = problem;
            this.Population = problem.Initialization();
            [~,this.consNum] = size(this.Population.cons);
            this.NP = problem.N;
            this.hvValue = 0;
            [~,this.FrontNo,this.CrowdDis] = EnvironmentalSelection(this.Population,problem.N,0);
        end
        
        function [reward, nextState, done, bestPop] = update(this,BOparameters,Problem)
            MatingPool = TournamentSelection(2,Problem.N,this.FrontNo,-this.CrowdDis);
            Offspring  = OperatorGA(Problem,this.Population(MatingPool));
            [this.Population,this.FrontNo,this.CrowdDis] = EnvironmentalSelection([this.Population,Offspring],Problem.N,BOparameters);
            
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
    end
end
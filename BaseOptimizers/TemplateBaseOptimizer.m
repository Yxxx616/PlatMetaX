classdef TemplateBaseOptimizer < BASEOPTIMIZER
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
        function init(Algorithm,problem)
            Algorithm.curProblem = problem;
            Algorithm.Population = problem.Initialization();
            [~,Algorithm.consNum] = size(Algorithm.Population.cons);
            Algorithm.NP = problem.N;
            Algorithm.hvValue = 0;
            [~,Algorithm.FrontNo,Algorithm.CrowdDis] = EnvironmentalSelection(Algorithm.Population,problem.N,0);
        end
        
        function [reward, nextState, done, bestPop] = update(baseOptimizer,BOparameters,Problem)
            MatingPool = TournamentSelection(2,Problem.N,baseOptimizer.FrontNo,-baseOptimizer.CrowdDis);
            Offspring  = OperatorGA(Problem,baseOptimizer.Population(MatingPool));
            [baseOptimizer.Population,baseOptimizer.FrontNo,baseOptimizer.CrowdDis] = EnvironmentalSelection([baseOptimizer.Population,Offspring],Problem.N,BOparameters);
            
            nextState = calState(baseOptimizer);
            nofinish = baseOptimizer.NotTerminated(baseOptimizer.Population);
            done = ~nofinish;
            curHV = Problem.CalMetric('HV',baseOptimizer.Population);
            if isnan(curHV)
                curHV = 0;
                reward = -1;
            elseif curHV - baseOptimizer.hvValue <=0
                reward = 0;
            else
                reward = 1;
            end
            baseOptimizer.hvValue = curHV;
            if done
                bestPop = baseOptimizer.Population;
            else
                bestPop = 0;
            end
        end
    end
end


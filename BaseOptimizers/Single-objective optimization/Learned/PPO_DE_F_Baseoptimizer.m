classdef PPO_DE_F_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none> <learned/none>
% USE DQN TO SELECT THE MUTATION STRATEGY OF Differential evolution
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
        curProblem
        consNum
        Population
        NP
        baseperformance
        
        CR
        F
    end
    methods
        function init(baseOptimizer,problem)
            %% Parameter setting
            [baseOptimizer.CR] = baseOptimizer.ParameterSet(0.9);
            
            %% Generate random population
            baseOptimizer.curProblem = problem;
            baseOptimizer.Population = problem.Initialization();
            
            [~,baseOptimizer.consNum] = size(baseOptimizer.Population.cons);
            baseOptimizer.NP = problem.N;
            baseOptimizer.baseperformance = min(baseOptimizer.Population.objs);
        end
        
        function [reward, nextState, done, bestPop] = update(baseOptimizer,BOparameters,Problem)
            MatingPool = TournamentSelection(2,2*Problem.N,FitnessSingle(baseOptimizer.Population));
            Offspring  = OperatorDE(Problem,baseOptimizer.Population,baseOptimizer.Population(MatingPool(1:end/2)),baseOptimizer.Population(MatingPool(end/2+1:end)),{baseOptimizer.CR,BOparameters,0,0});
            replace             = FitnessSingle(baseOptimizer.Population) > FitnessSingle(Offspring);
            baseOptimizer.Population(replace) = Offspring(replace);
            nextState = calSOPState(baseOptimizer);
            nofinish = baseOptimizer.NotTerminated(baseOptimizer.Population);
            done = ~nofinish;
            curBP = min(baseOptimizer.Population.objs);
            if isnan(curBP)
                curBP = baseOptimizer.baseperformance;
                reward = -1;
            elseif curBP - baseOptimizer.baseperformance >=0
                reward = 0;
            else
                reward = 1;
            end
            baseOptimizer.baseperformance = curBP;
            if done
                bestPop = baseOptimizer.Population;
            else
                bestPop = 0;
            end
        end
    end
end
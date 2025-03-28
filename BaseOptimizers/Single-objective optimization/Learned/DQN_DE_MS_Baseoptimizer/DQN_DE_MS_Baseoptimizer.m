classdef DQN_DE_MS_Baseoptimizer < BASEOPTIMIZER
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
        curBP
        bestInd
        
        CR
        F
    end
    methods
        function init(baseOptimizer,problem)
            %% Parameter setting
            [baseOptimizer.CR,baseOptimizer.F] = baseOptimizer.ParameterSet(0.9,0.5);
            
            %% Generate random population
            baseOptimizer.curProblem = problem;
            baseOptimizer.Population = problem.Initialization();
            
            [~,baseOptimizer.consNum] = size(baseOptimizer.Population.cons);
            baseOptimizer.NP = problem.N;
            [baseOptimizer.baseperformance,idx] = min(baseOptimizer.Population.objs);
            baseOptimizer.curBP = baseOptimizer.baseperformance;
            baseOptimizer.bestInd = baseOptimizer.Population(idx);
        end
        
        function [reward, nextState, done, bestPop] = update(baseOptimizer,BOparameters,Problem)
            MatingPool = TournamentSelection(2,2*Problem.N,baseOptimizer.Population.objs);
            Offspring  = LearnedMSOperatorDE(Problem,baseOptimizer.Population,baseOptimizer.Population(MatingPool(1:end/2)),baseOptimizer.Population(MatingPool(end/2+1:end)),baseOptimizer.bestInd, BOparameters,{baseOptimizer.CR,baseOptimizer.F,0,0});
            replace             = FitnessSingle(baseOptimizer.Population) > FitnessSingle(Offspring);
            baseOptimizer.Population(replace) = Offspring(replace);
            nextState = calSOPState(baseOptimizer);
            nofinish = baseOptimizer.NotTerminated(baseOptimizer.Population);
            done = ~nofinish;
            [currentBP,bestindIdx] = min(baseOptimizer.Population.objs);
            if isnan(currentBP)
                currentBP = baseOptimizer.curBP;
                reward = -10;
            elseif currentBP - baseOptimizer.curBP > 0
                reward = -1;
            elseif currentBP - baseOptimizer.curBP == 0
                reward  = 0;
            else
                reward = 1;
            end
            baseOptimizer.curBP = currentBP;
            if currentBP < baseOptimizer.baseperformance
                baseOptimizer.baseperformance = min(currentBP,baseOptimizer.baseperformance);
                baseOptimizer.bestInd = baseOptimizer.Population(bestindIdx);
            end
            if baseOptimizer.baseperformance > 1e-8
                reward = reward - 1;
            end
            if done
                bestPop = baseOptimizer.Population;
                if baseOptimizer.baseperformance < 1e-8
                    reward = reward + 100;
                end
            else
                bestPop = 0;
            end
        end
    end
end
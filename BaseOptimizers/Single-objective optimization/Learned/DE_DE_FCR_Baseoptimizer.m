classdef DE_DE_FCR_Baseoptimizer < BASEOPTIMIZER
% <1997> <single> <real/integer> <large/none> <constrained/none> <learned/none>
% Differential evolution
% CR --- 0.9 --- Crossover constant
% F  --- 0.5 --- Mutation factor

%------------------------------- Reference --------------------------------
% R. Storn and K. Price, Differential evolution-a simple and efficient
% heuristic for global optimization over continuous spaces, Journal of
% Global Optimization, 1997, 11(4): 341-359.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    properties
        curProblem
        consNum
        Population
        NP
        baseperformance
        curBP
        
        CR
        F
    end
    methods
        function init(baseOptimizer,problem)     
            %% Generate random population
            baseOptimizer.curProblem = problem;
            baseOptimizer.Population = problem.Initialization();
            
            [~,baseOptimizer.consNum] = size(baseOptimizer.Population.cons);
            baseOptimizer.NP = problem.N;
            baseOptimizer.baseperformance = min(baseOptimizer.Population.objs);
            baseOptimizer.curBP = baseOptimizer.baseperformance;
        end
        
        function [reward, nextState, done, bestPop] = update(baseOptimizer,BOparameters,Problem)
            while baseOptimizer.NotTerminated(baseOptimizer.Population)
                MatingPool = TournamentSelection(2,2*Problem.N,FitnessSingle(baseOptimizer.Population));
                Offspring  = OperatorDE(Problem,baseOptimizer.Population,baseOptimizer.Population(MatingPool(1:end/2)),baseOptimizer.Population(MatingPool(end/2+1:end)),{BOparameters(1),BOparameters(2),0,0});
                replace             = FitnessSingle(baseOptimizer.Population) > FitnessSingle(Offspring);
                baseOptimizer.Population(replace) = Offspring(replace);
            end
            nextState = 0;
            reward = -min(baseOptimizer.Population.objs);
            bestPop = baseOptimizer.Population;
            done = true;
        end
        
        function state = calCurProblemState(this)
            pname = class(this.curProblem);
            state = regexp(pname, '\d+', 'match');
        end
    end
end
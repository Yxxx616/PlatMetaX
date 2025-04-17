classdef Test < handle
    properties
        MetaOptimizer
        moName
        BaseOptimizer
        boName
        TestingSet
        TestingSetName
        env
    end
    methods
        function obj = Test(moName, BO, envName, problemset)
            obj.moName = moName;
            if isa(BO, 'handle')
                obj.BaseOptimizer = BO;
                obj.boName = class(BO);
            else
                obj.boName = BO;
                obj.BaseOptimizer = feval(BO);
            end
            if isa(problemset,'struct')
                obj.TestingSetName = problemset.psName;
                [~, obj.TestingSet] = splitProblemSet(problemset);
            elseif isa(problemset, 'handle')
                obj.TestingSetName = class(problemset);
                obj.TestingSet{1} = problemset;
            end
            obj.env = feval(envName, obj.TestingSet, obj.BaseOptimizer, 'test');
            fn = load(['AgentModel/', moName, '_finalAgent.mat']);
            obj.MetaOptimizer = fn.saved_agent;
        end
        
        function results = run(obj)
            set(0, 'DefaultFigureVisible', 'on');
            simOpts = rlSimulationOptions('NumSimulations',length(obj.TestingSet)); 
            testingInfo = sim(obj.env,obj.MetaOptimizer,simOpts);
            results.testingInfo = testingInfo;
        end
    end
end
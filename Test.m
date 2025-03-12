classdef Test < handle
    properties
        MetaOptimizer
        BaseOptimizer
        TestingSet
        TestingSetName
        env
    end
    methods
        function obj = Test(mo, bo, envConfig, problemset)
            obj.BaseOptimizer = bo();
            if isa(problemset,'struct')
                obj.TestingSetName = problemset.psName;
                [~, obj.TestingSet] = splitProblemSet(problemset);
            elseif isa(problemset, 'handle')
                obj.TestingSetName = class(problemset);
                obj.TestingSet{1} = problemset;
            end
            obj.env = envConfig(obj.TestingSet,obj.BaseOptimizer,'test');
            fn = load(['AgentModel/', functions(mo).function, '_finalAgent.mat']);
            obj.MetaOptimizer = fn.agent;
        end
        
        function results = run(obj)
            simOpts = rlSimulationOptions('NumSimulations',length(obj.TestingSet)); 
            testingInfo = sim(obj.env,obj.MetaOptimizer,simOpts);
%             bestPops = obj.env.getBestPops();
            results.testingInfo = testingInfo;
%             results.bestPops = bestPops;
        end
    end
end
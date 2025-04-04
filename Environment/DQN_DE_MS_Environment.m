classdef DQN_DE_MS_Environment < rl.env.MATLABEnvironment
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        problemSet
        curPIdx
        curProblem
        baseoptimizer
        bestPops
        task
    end
    
    properties
        State
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end
    
    methods
        function this = DQN_DE_MS_Environment(ps,bo,task)
            % Initialize observation settings  /population state
            ObservationInfo = rlNumericSpec([13 1]);
            ObservationInfo.Name = 'observations';
            ObservationInfo.Description = '';

            % Initialize BOparameters settings    /
            ActionInfo = rlFiniteSetSpec([1, 2, 3, 4]);
            ActionInfo.Name = 'BOparameters';
            ActionInfo.Description = 'DE-Mutation strategy selection';

            % The following line implements built-in functions of the RL environment
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.problemSet = ps;
            this.baseoptimizer = bo;
            this.bestPops = containers.Map;
            this.task = task;
            this.curPIdx = 1;
        end
        
        function InitialObservation = reset(this)
            if this.curPIdx > length(this.problemSet)
                this.curPIdx = 1;
            end
            this.curProblem = this.problemSet{this.curPIdx};
            this.baseoptimizer.Init(this.curProblem)
            if this.curProblem.M > 1
                InitialObservation = calMOPState(this.baseoptimizer);
            else
                InitialObservation = calSOPState(this.baseoptimizer);
            end
            this.State = InitialObservation;
        end

        function [Observation,Reward,IsDone,LoggedSignals] = step(this, BOparameters)
            LoggedSignals = [];
             [Reward, Observation, IsDone, bestPop] = this.baseoptimizer.update(BOparameters, this.curProblem);
            this.State = Observation;
            this.IsDone = IsDone;
            if IsDone
                this.bestPops(class(this.curProblem)) = bestPop;
            end
        end
        
        function bestPops = getBestPops(this)
            bestPops=this.bestPops;
        end
    end
end
classdef DQN_DE_AS_Environment < rl.env.MATLABEnvironment
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
        IsDone = false        
    end
    
    methods
        function this = DQN_DE_AS_Environment(ps,bo,task)
            % Initialize observation settings  /population state
            ObservationInfo = rlNumericSpec([12 1]);
            ObservationInfo.Name = 'observations';
            ObservationInfo.Description = '';

            % Initialize BOparameters settings    /
            ActionInfo = rlFiniteSetSpec([1, 2, 3, 4, 5, 6]);
            ActionInfo.Name = 'BOparameters';
            ActionInfo.Description = 'DE-selection';

            % The following line implements built-in functions of the RL environment
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.problemSet = ps;
            this.baseoptimizer = bo;
            this.bestPops = containers.Map;
            this.task = task;
            this.curPIdx = 1;
        end
        
        function InitialObservation = reset(this)
            this.curPIdx = randi(length(this.problemSet));
            this.curProblem = this.problemSet{this.curPIdx};
            this.baseoptimizer.Init(this.curProblem);
            InitialObservation = this.baseoptimizer.calState();
            this.State = InitialObservation;
        end

        function [Observation,Reward,IsDone,LoggedSignals] = step(this, BOparameters)
            LoggedSignals = [];
             [Reward, ~, IsDone, bestPop] = this.baseoptimizer.update(BOparameters, this.curProblem);
            Observation = this.State;
            this.IsDone = IsDone;
            if IsDone && strcmp(this.task, 'test')
                this.bestPops(class(this.curProblem)) = bestPop;
            end
        end
        
        function bestPops = getBestPops(this)
            bestPops=this.bestPops;
        end
    end
end


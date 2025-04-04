classdef TemplateEnvironment < rl.env.MATLABEnvironment
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        % Specify and initialize environment's necessary properties    
        curProblem
        baseoptimizer
        problemSet
    end
    
    properties
        State
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = TemplateEnvironment(ps, bo)
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'Features';
            ObservationInfo.Description = 'X,X,X,X';
            
            % Initialize Action settings    Continuous
            ActionInfo = rlFiniteSetSpec([-1 1]);
            ActionInfo.Name = 'Base-optimizer Parameters (Type and Range)';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.problemSet = ps;
            this.baseoptimizer = bo;
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
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
            this.curProblemState = InitialObservation;
        end

        function [Observation,Reward,IsDone,LoggedSignals] = step(this, BOparameters)
            %METHOD1 此处显示有关此方法的摘要
            %   此处显示详细说明
            LoggedSignals = [];
             [Reward, Observation, IsDone, bestPop] = this.baseoptimizer.update(BOparameters, this.curProblem);
            this.curProblemState = Observation;
            this.IsDone = IsDone;
            if IsDone
                this.bestPops(class(this.curProblem)) = bestPop;
                if strcmp(this.task, 'test')
                    saveTestResults(this.baseoptimizer, this.curProblem);
                end
                this.curPIdx = this.curPIdx + 1;
            end
        end
    end
end

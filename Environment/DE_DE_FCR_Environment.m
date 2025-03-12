classdef DE_DE_FCR_Environment < rl.env.MATLABEnvironment
    %MYENVIRONMENT: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        curProblem
        bestPop % best population of curProblem
        baseoptimizer
        problemSet
        curPIdx
        indCount
        task
        metaNP = 10 %need to set the same with corresponding meta-optimizer
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = DE_DE_FCR_Environment(ps, bo, task)
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([1 1], 'LowerLimit', 1, 'UpperLimit', 24); %the UpperLimit should be set to length(ps)
            ObservationInfo.Name = 'ProblemIdx';
            ObservationInfo.Description = 'F_';
            
            % Initialize Action settings    Continuous
            ActionInfo = rlNumericSpec([2 1],'LowerLimit', 0.5, 'UpperLimit', 1);
            ActionInfo.Name = 'Base-optimizer Parameters (Type and Range)';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.problemSet = ps;
            this.bestPop = containers.Map;
            this.baseoptimizer = bo;
            this.curPIdx = 1;
            this.task = task;
            this.indCount = 0;
        end
        function metanp = getmetanp(this)
            metanp = this.metaNP;
        end
        function InitialObservation = reset(this)
            if this.curPIdx > length(this.problemSet)
                this.curPIdx = 1;
            end
            this.curProblem = this.problemSet{this.curPIdx};
            this.baseoptimizer.Init(this.curProblem)
            InitialObservation = this.curPIdx;
        end
       
        function [Observation,Reward,IsDone,LoggedSignals] = step(this, BOparameters)
            if strcmp(this.task, 'test')
                [Reward, ~, ~, bestpop] = this.baseoptimizer.update(BOparameters, this.curProblem);
                IsDone = true;
                this.bestPop(class(this.curProblem)) = bestpop;
                LoggedSignals = [];
                this.curPIdx = this.curPIdx + 1;
                this.IsDone = IsDone;
                Observation = str2double(this.baseoptimizer.calCurProblemState());
                saveTestResults(this.baseoptimizer, this.curProblem);
            else
                IsDone = false;
                [Reward, ~, ~, ~] = this.baseoptimizer.update(BOparameters, this.curProblem);
                this.indCount = this.indCount + 1;


                if this.indCount >= this.metaNP
                    this.curPIdx = this.curPIdx + 1;
                    this.indCount = 0;
                end


                if this.curPIdx > length(this.problemSet)
                    IsDone = true;
                    this.curProblem = this.problemSet{1}; 
                else
                    this.curProblem = this.problemSet{this.curPIdx};
                end
                this.baseoptimizer.Init(this.curProblem);
                LoggedSignals = [];
                this.IsDone = IsDone;
                Observation = str2double(this.baseoptimizer.calCurProblemState());
            end
        end
        function bestPops = getBestPops(this)
            bestPops=this.bestPop;
        end
    end
end

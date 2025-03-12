classdef MLP_Alg_Rec_Environment < rl.env.MATLABEnvironment
    %MYENVIRONMENT: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        curProblem
        baseoptimizer
        problemSet
        curPIdx
        task
        algCount
        bestPop
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
        function this = MLP_Alg_Rec_Environment(ps, bo, task)
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([17 1]);
            ObservationInfo.Name = 'ProblemIdx';
            ObservationInfo.Description = 'F_';
            
            % Initialize Action settings    Continuous
            ActionInfo = rlFiniteSetSpec([1, 2, 3, 4, 5]);
            ActionInfo.Name = 'Algorithm recommendation';
            ObservationInfo.Description = 'ABC,CSO,DE,PSO,SA';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.problemSet = ps;
            this.baseoptimizer = bo;
            this.curPIdx = 1;
            this.task = task;
            this.algCount = 0;
            
        end
        function InitialObservation = reset(this)
            if this.curPIdx > length(this.problemSet)
                this.curPIdx = 1;
            end
            this.curProblem = this.problemSet{this.curPIdx};
            this.baseoptimizer.Init(this.curProblem)
            InitialObservation = this.curProblem.extractFeatures();
        end
       
        function [Observation,Reward,IsDone,LoggedSignals] = step(this, algidx)
            if strcmp(this.task, 'test')
                [Reward, ~, ~, bestpop] = this.baseoptimizer.update(algidx, this.curProblem);
                IsDone = true;
                saveTestResults(this.baseoptimizer, this.baseoptimizer.pro);
                this.bestPop = bestpop;
                this.curPIdx = this.curPIdx + 1;

                LoggedSignals = [];
                this.IsDone = IsDone;
                Observation = this.curProblem.instanceFeatures;
            else
                set(0, 'DefaultFigureVisible', 'off');
                IsDone = false;
                [Reward, ~, ~, ~] = this.baseoptimizer.update(algidx, this.curProblem);
                this.algCount = this.algCount + 1;


                if this.algCount >= 5
                    this.curPIdx = this.curPIdx + 1;
                    this.algCount = 0;
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
                Observation = this.curProblem.instanceFeatures;

            end
        end
        function bestPops = getBestPops(this)
            bestPops=this.bestPop;
        end
    end
end

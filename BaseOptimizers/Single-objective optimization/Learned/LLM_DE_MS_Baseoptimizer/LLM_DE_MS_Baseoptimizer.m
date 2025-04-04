classdef LLM_DE_MS_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none> <learned/none>

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        consNum
        Population
        NP
        baseperformance
        curBP
    end
    methods
        function init(this,problem)     
            %% Generate random population
            this.pro = problem;
            this.Population = problem.Initialization();
            
            [~,this.consNum] = size(this.Population.cons);
            this.NP = problem.N;
            this.baseperformance = min(this.Population.objs);
            this.curBP = this.baseperformance;
        end
        function [reward, nextState, done, bestPop] = update(this,updatefuncidx,Problem)
            funcname = ['updateFunc', num2str(updatefuncidx)];
            errormsg = '';
            try
                while this.NotTerminated(this.Population)
                    Offspring = feval(funcname, this.Population.decs, this.Population.objs, sum(this.Population.cons, 2));
                    Offspring = Problem.Evaluation(Offspring);
                    replace             = FitnessSingle(this.Population) > FitnessSingle(Offspring);
                    this.Population(replace) = Offspring(replace);
                end
                reward = this.baseperformance -min(this.Population.objs);
            catch ME
                while this.NotTerminated(this.Population)
                    MatingPool = TournamentSelection(2,2*Problem.N,FitnessSingle(this.Population));
                    Offspring  = OperatorDE(Problem,this.Population,this.Population(MatingPool(1:end/2)),this.Population(MatingPool(end/2+1:end)),{0.9,0.5,0,0});
                    replace             = FitnessSingle(this.Population) > FitnessSingle(Offspring);
                    this.Population(replace) = Offspring(replace);
                end
                reward = -Inf;
                errormsg = ['The generated Matlab code cannot run successfully. Matlab returns:' ME.message];
            end
            nextState = 0;
            bestPop = this.Population;
            done = true;
%             results = struct(...
%                 'iter', updatefuncidx,...
%                 'metrics', struct(...
%                     'MinValue', 0.843,...
%                     'FeasibleRate', 0.0098,...
%                     'time', this.metric.runtime),...
%                 'error', errormsg);
%             py.testdeepseek.report_results(results,updatefuncidx);
%             appendToHistory(results);
        end
        
        function state = calCurProblemState(this)
            pname = class(this.pro);     
            pidx = regexp(pname, 'F\d+', 'match');   
            flatdec = reshape(this.Population.decs,1,[]);
            objs = this.Population.objs;
            cv = sum(this.Population.cons,2);
            state = [flatdec objs' cv' str2double(pidx{1}(2:end))];
        end
    end
end


% 定义超时异常处理函数
function throwTimeoutException(src, event, ME)
    error('TimeoutException: The code execution exceeded the allowed time.');
end
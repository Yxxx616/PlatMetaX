classdef DDPG_DE_F_Environment < rl.env.MATLABEnvironment
    %ENVIRONMENT 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        problemSet
        curPIdx
        curProblem
        baseoptimizer
        bestPops
        task
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State = zeros(13,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end
    
    methods
        function this = DDPG_DE_F_Environment(ps,bo,task)
            % Initialize observation settings  /population state
            ObservationInfo = rlNumericSpec([13 1]);
            ObservationInfo.Name = 'observations';
            ObservationInfo.Description = '';

            % Initialize BOparameters settings    /
            ActionInfo = rlNumericSpec([1 1],'LowerLimit',0,'UpperLimit',2);
            ActionInfo.Name = 'BOparameters';
            ActionInfo.Description = 'DE-F ONLINE TUNING';

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
            %METHOD1 此处显示有关此方法的摘要
            %   此处显示详细说明
            LoggedSignals = [];
             [Reward, Observation, IsDone, bestPop] = this.baseoptimizer.update(BOparameters, this.curProblem);
            this.State = Observation;
            this.IsDone = IsDone;
            if IsDone
                this.bestPops(class(this.curProblem)) = bestPop;
                if strcmp(this.task, 'test')
                    saveTestResults(this.baseoptimizer, this.curProblem);
                end
                this.curPIdx = this.curPIdx + 1;
            end
        end
        
        function bestPops = getBestPops(this)
            bestPops=this.bestPops;
        end
    end
end


function saveTestResults(Algorithm,Problem)
% The default output function of BASEOPTIMIZER

    clc; fprintf('%s on %d-objective %d-variable %s (%6.2f%%), %.2fs passed...\n',class(Algorithm),Problem.M,Problem.D,class(Problem),Problem.FE/Problem.maxFE*100,Algorithm.metric.runtime);
    
    if Problem.FE >= Problem.maxFE
        if Algorithm.save < 0
            if isempty(Algorithm.metName)
                if Problem.M == 1
                    Algorithm.metName = {'Min_value','Feasible_rate'};
                elseif length(Algorithm.result{end}) >= size(Problem.optimum,1)
                    Algorithm.metName = {'HV','Feasible_rate'};
                else
                    Algorithm.metName = {'IGD','HV','GD','Feasible_rate'};
                end
            elseif ~iscell(Algorithm.metName)
                Algorithm.metName = {Algorithm.metName};
            end
            value = Algorithm.CalMetric(Algorithm.metName{1});
            figure('NumberTitle','off','Name',sprintf('%s : %.4e  Runtime : %.2fs',Algorithm.metName{1},value(end),Algorithm.CalMetric('runtime')));
            title(sprintf('%s on %s',class(Algorithm),class(Problem)),'Interpreter','none');
            top = uimenu(gcf,'Label','Data source');
            if Problem.M > 1
                uimenu(top,'Label','Population (obj.)','CallBack',{@(h,~,Pro,P)eval('Draw(gca);Pro.DrawObj(P);cb_menu(h);'),Problem,Algorithm.result{end}});
            end
            uimenu(top,'Label','Population (dec.)','CallBack',{@(h,~,Pro,P)eval('Draw(gca);Pro.DrawDec(P);cb_menu(h);'),Problem,Algorithm.result{end}});
            if Problem.M > 1
                uimenu(top,'Label','True Pareto front','CallBack',{@(h,~,P)eval('Draw(gca);Draw(P,{''\it f\rm_1'',''\it f\rm_2'',''\it f\rm_3''});cb_menu(h);'),Problem.optimum});
            end
            cellfun(@(s)uimenu(top,'Label',s,'CallBack',{@(h,~,A)eval('Draw(gca);Draw([cell2mat(A.result(:,1)),A.CalMetric(h.Label)],''-k.'',''LineWidth'',1.5,''MarkerSize'',10,{''Number of function evaluations'',strrep(h.Label,''_'','' ''),[]});cb_menu(h);'),Algorithm}),Algorithm.metName);
            set(top.Children(length(Algorithm.metName)),'Separator','on');
            top.Children(end).Callback{1}(top.Children(end),[],Problem,Algorithm.result{end});
        elseif Algorithm.save > 0
            for i = 1 : length(Algorithm.metName)
                Algorithm.CalMetric(Algorithm.metName{i});
            end
            result = Algorithm.result;
            metric = Algorithm.metric;
            folder = fullfile('Data',class(Algorithm));
            [~,~]  = mkdir(folder);
            file   = fullfile(folder,sprintf('%s_%s_M%d_D%d_',class(Algorithm),class(Problem),Problem.M,Problem.D));
            if isempty(Algorithm.run) 
                Algorithm.run = 1;
                while exist([file,num2str(Algorithm.run),'.mat'],'file') == 2
                    Algorithm.run = Algorithm.run + 1;
                end
            end
            save([file,num2str(Algorithm.run),'.mat'],'result','metric');
        end
    end
end

function cb_menu(h)
% Switch between the selected menu

    set(get(get(h,'Parent'),'Children'),'Checked','off');
    set(h,'Checked','on');
end
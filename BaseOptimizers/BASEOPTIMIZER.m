classdef BASEOPTIMIZER < handle & matlab.mixin.Heterogeneous
    properties
        parameter = {};                 % Parameters of the algorithm
        save      = -10;            	% Number of populations saved in an execution
        run       = [];                	% Current execution number
        metName   = {};                 % Names of metrics to calculate
        outputFcn = @DefaultOutput;     % Function called after each generation
        pro;                            % Problem solved in current execution
        result;                         % Populations saved in current execution
        metric;                         % Metric values of current populations
        starttime;                      % Used for runtime recording
    end
    methods(Access = protected)
        function obj = BASEOPTIMIZER(varargin)
            isStr = find(cellfun(@ischar,varargin(1:end-1))&~cellfun(@isempty,varargin(2:end)));
            for i = isStr(ismember(varargin(isStr),{'parameter','save','run','metName','outputFcn'}))
                obj.(varargin{i}) = varargin{i+1};
            end
        end
    end
    methods(Sealed)
        function Init(obj,Problem)
        %Solve - Use the algorithm to solve a problem.
        %
        %   obj.Solve(Pro) uses the algorithm to solve a problem, where Pro
        %   is a PROBLEM object.
        %
        %   In terms of the default obj.outputFcn, the result will be
        %   displayed when obj.save = 0 and saved when obj.save > 0.
        %
        %   Example:
        %       Algorithm.Solve(Problem)
            
            try
                obj.result = {};
                obj.metric = struct('runtime',0);
                obj.pro    = Problem;
                obj.pro.FE = 0;
                addpath(fileparts(which(class(obj))));
                addpath(fileparts(which(class(obj.pro))));
                obj.starttime = tic;
                obj.init(obj.pro);
            catch err
                if ~strcmp(err.identifier,'PlatMetaBBO:Termination')
                    rethrow(err);
                end
            end
        end
    end
    methods
        function init(obj,Problem)
        end
    end
    methods(Access = protected, Sealed)
        function nofinish = NotTerminated(obj,Population)
        %NotTerminated - The function called after each generation of the
        %execution.
        %
        %   obj.NotTerminated(P) stores the population P as the result of
        %   the current execution, and returns true if the algorithm should
        %   be terminated, i.e., the number of function evaluations or
        %   runtime exceeds.
        %
        %   obj.outputFcn is called here, whose runtime will not be counted
        %   in the runtime of current execution.
        %
        %   Example:
        %       while Algorithm.NotTerminated(Population)
        %           ... ...
        %       end
        
            obj.metric.runtime = obj.metric.runtime + toc(obj.starttime);
            if obj.pro.maxRuntime < inf
                obj.pro.maxFE = obj.pro.FE*obj.pro.maxRuntime/obj.metric.runtime;
            end
            num   = max(1,abs(obj.save));
            index = max(1,min(min(num,size(obj.result,1)+1),ceil(num*obj.pro.FE/obj.pro.maxFE)));
            obj.result(index,:) = {obj.pro.FE,Population};
            drawnow('limitrate');
            obj.outputFcn(obj,obj.pro);
            nofinish = obj.pro.FE < obj.pro.maxFE;
            obj.starttime = tic;
        end
        function varargout = ParameterSet(obj,varargin)
        %ParameterSet - Obtain the parameters of the algorithm.
        %
        %   [p1,p2,...] = obj.ParameterSet(v1,v2,...) sets the values of
        %   parameters p1, p2, ..., where each parameter is set to the
        %   value given in obj.parameter if obj.parameter is specified, and
        %   set to the value given in v1, v2, ... otherwise.
        %
        %   Example:
        %       [p1,p2,p3] = Algorithm.ParameterSet(1,2,3)

            varargout = varargin;
            specified = ~cellfun(@isempty,obj.parameter);
            varargout(specified) = obj.parameter(specified);
        end
    end
    methods(Sealed)
        function Scores = CalMetric(obj,metName)
        % Calculate metric values
        
            if ~isfield(obj.metric,metName)
                obj.metric.(metName) = cellfun(@(S)obj.pro.CalMetric(metName,S),obj.result(:,2));
            end
            Scores = obj.metric.(metName);
        end
        
        function hvvalue = CalHV(obj)
%         Calculate metric values
            hvvalue = cellfun(@(S)obj.pro.CalMetric('HV',S),obj.result(:,2));
        end
    end
end

function DefaultOutput(Algorithm,Problem)
% The default output function of BASEOPTIMIZER no figure;
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

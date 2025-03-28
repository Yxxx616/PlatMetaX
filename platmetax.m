function varargout = platmetax(varargin)
%platmetax - The main function of PlatMetaX.
%
%   platmetax() displays the GUI of PlatMetaX.
%
%   platmetax('Name',Value,'Name',Value,...) runs an algorithm on a problem
%   with specified parameter settings.
%
% All the acceptable names and values are:
%   'metabboComps'  <String>         common str of MetaBBO env, MO, BO
%	'algorithm'     <function handle>	an algorithm
%   'problem set'   <String>	        a problem set
%	'problem'       <function handle>	a problem
%   'N'             <positive integer>  population size
%   'M'             <positive integer>  number of objectives
%   'D'             <positive integer>  number of variables
%	'maxFE'         <positive integer>  maximum number of function evaluations
%   'maxRuntime'    <positive integer>  maximum runtime (in second)
%   'save'       	<integer>           number of saved populations
%   'run'           <integer>           current run number
%   'metName'       <string>            names of metrics to calculate
%   'outputFcn'     <function handle>   function called after each iteration
%   'encoding'      <string>            encoding scheme of each decision variable (1.real 2.integer 3.label 4.binary 5.permutation)
%   'lower'         <vector>            lower bound of each decision variable
%   'upper'         <vector>            upper bound of each decision variable
%   'initFcn'       <function handle>   function for initializing solutions
%   'evalFcn'       <function handle>   function for evaluating solutions
%   'decFcn'        <function handle>   function for repairing invalid solutions
%   'objFcn'        <function handle>   objective functions
%   'conFcn'        <function handle>   constraint functions
%   'gradFcn'       <function handle>   function for calculating the gradients of objectives and constraints
%   'data'          <any>               data of the problem
%   'once'          <logical>           whether the inputs of evalFcn, decFcn, objFcn, conFcn can be multiple solutions
%
%   Example:
%
%       platmetax()
%
%   displays the GUI of PlatMetaX.
%
%%  Train meta-optimizer
%   platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB')
%   platmetax('task', @Train, 'metabboComps', 'DQN_DE_MS', 'problemSet','BBOB')
%   platmetax('task', @Train, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB','N',50,'D',10 )
%   platmetax('task', @Train, 'metabboComps', 'DE_DE_FCR', 'problemSet','BBOBEC')
%   platmetax('task', @Train, 'metabboComps', 'MLP_Alg_Rec', 'problemSet','TSPs')
%   platmetax('task', @Train, 'metabboComps', 'DQN_NSGAII_MSDTLZ', 'problemSet','DTLZ')
%   platmetax('task', @Train, 'metabboComps', 'DDPG_NSGAII_Alpha', 'problemSet','MW')
%   platmetax('task', @Train, 'metabboComps', 'LLM_DE_MS', 'problemSet','CEC2017LLM','N',50,'D',10 )
%%  Test meta-optimizer
%   1.测试RL-based meta-optimizer：DDPG_DE_F（实现使用DDPG在线调整DE参数F）
%   platmetax('task', @Test, 'metabboComps', 'DDPG_DE_F', 'problemSet','BBOB') 
%   测试RL-based meta-optimizer：DQN_DE_MS（实现使用DQN在线调整DE的变异策略MS）
%   platmetax('task', @Test, 'metabboComps', 'DQN_DE_MS', 'problemSet','BBOB')
%   2.测试SL-based meta-optimizer：MLP_Alg_Rec（实现使用MLP神经网络对不同TSP示例算法推荐ABC/CSO/DE/PSO/SA）
%   platmetax('task', @Test, 'metabboComps', 'MLP_Alg_Rec', 'problemSet','TSPs')
%   3.测试EC-based meta-optimizer：DE_DE_FCR（实现使用DE离线调整DE参数F和CR）
%   platmetax('task', @Test, 'metabboComps', 'DE_DE_FCR', 'problemSet','BBOBEC')
%%  Test Traditional base-optimizer
%   platmetax('task', @TestTraditionalAlg,'algorithm',@PSO,'problem',@SOP_F1,'N',50,'maxFE',20000)

    cd(fileparts(mfilename('fullpath')));
    addpath(genpath(cd));
    rng('shuffle');
    if isempty(varargin)
        if verLessThan('matlab','9.9')
            errordlg('Fail to create the GUI of PlatMetaX since the version for MATLAB is lower than R2020b. You can use PlatMetaX without GUI by calling platmetax() with parameters.','Error','modal');
        else
            try
                platmetaxGUI();
            catch err
                errordlg('Fail to create the GUI, please make sure all the folders of PlatMetaX have been added to search path.','Error','modal');
                rethrow(err);
            end
        end
    else
        if verLessThan('matlab','9.4')
            error('Fail to use platmetax since the version for MATLAB is lower than R2018a. Please update your MATLAB software.');
        else
            isStr = find(cellfun(@ischar,varargin(1:end-1))&~cellfun(@isempty,varargin(2:end)));
            index = isStr(find(strcmp(varargin(isStr),'task'),1)) + 1;
            if isempty(index)
                error('Fail to use platmetax since there is no correct task!');
            else
                taskName = varargin{index};
            end
            if isequal(taskName, @TestTraditionalAlg)
                [PRO,input] = getSetting(varargin);
                Problem     = PRO(input{:});
                [ALG,input] = getSetting(varargin,Problem);
                if nargout > 0
                    Algorithm = ALG(input{:},'save',0);
                else
                    Algorithm = ALG(input{:});
                end
                Algorithm.Solve(Problem);
                if nargout > 0
                    P = Algorithm.result{end};
                    varargout = {P.decs,P.objs,P.cons};
                end
            else
                [metaOptimizer, baseOptimizer,env, problemSet] = getConfig(varargin);
                task = taskName(metaOptimizer, baseOptimizer, env, problemSet);
                result = task.run(); 
                close all;
            end
        end
    end
end

function [mo, bo, env, problemSet] = getConfig(Setting)
    isStr = find(cellfun(@ischar,Setting(1:end-1))&~cellfun(@isempty,Setting(2:end)));
    index = isStr(find(strcmp(Setting(isStr),'metabboComps'),1)) + 1;
    if isempty(index)
        error('Fail to use platmetax since there is no correct metabbo!');
    else
        metabboComponents  = Setting{index};
        env = str2func([metabboComponents '_Environment']);
        mo = str2func([metabboComponents '_Metaoptimizer']);
        bo = str2func([metabboComponents '_Baseoptimizer']);
%         Setting = [Setting,{'parameter'},{Setting{index}(2:end)}];
    end
    
    index = isStr(find(strcmp(Setting(isStr),'problemSet'),1)) + 1;
    if isempty(index)
        error('Fail to use platmetax since there is no problems!');
    elseif iscell(Setting{index})
        problemSet.psName = Setting{index}{1};
        problemSet.pSetting = [Setting,{'parameter'},{Setting{index}(2:end)}];
    else
        problemSet.psName = Setting{index};
        problemSet.pSetting = Setting; % all input parameters
    end
end

function [name,Setting] = getSetting(Setting,Pro)
    isStr = find(cellfun(@ischar,Setting(1:end-1))&~cellfun(@isempty,Setting(2:end)));
    if nargin > 1
        index = isStr(find(strcmp(Setting(isStr),'algorithm'),1)) + 1;
        if isempty(index)
            names = {@BSPGA,@GA,@SACOSO,@GA;@PMMOEA,@NSGAIII,@KRVEA,@NSGAIII;@RVEA,@RVEA,@CSEA,@RVEA};
            name  = names{find([Pro.M<2,Pro.M<4,1],1),find([all(Pro.encoding==4),any(Pro.encoding>2),Pro.maxFE<=1000&Pro.D<=10,1],1)};
        elseif iscell(Setting{index})
            name    = Setting{index}{1};
            Setting = [Setting,{'parameter'},{Setting{index}(2:end)}];
        else
            name = Setting{index};
        end
    else
        index = isStr(find(strcmp(Setting(isStr),'problem'),1)) + 1;
        if isempty(index)
            name = @UserProblem;
        elseif iscell(Setting{index})
            name    = Setting{index}{1};
            Setting = [Setting,{'parameter'},{Setting{index}(2:end)}];
        else
            name = Setting{index};
        end
    end
end
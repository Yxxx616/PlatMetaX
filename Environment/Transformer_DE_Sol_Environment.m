classdef Transformer_DE_Sol_Environment < rl.env.MATLABEnvironment
    properties
        BaseOptimizer    % 基础优化器实例
        Problem          % cur problem
        ProblemSet       % ProblemSet
        StateEncoder     % 状态编码器
        
    end
    
    methods
        function obj = Transformer_DE_Sol_Environment(ps, bo, task)
            % 初始化观测和动作空间
            maxTreeDepth = 3;
            problem = ps{1};
            ObservationInfo = rlNumericSpec([problem.D+2, 1]);
            ActionInfo = rlNumericSpec([problem.D, 1],...
                'LowerLimit',0,'UpperLimit',1);
            % 调用父类构造函数
            obj = obj@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            % 创建基础优化器
            obj.BaseOptimizer = bo();
            obj.Problem = problem;
            obj.ProblemSet = ps;
        end
        
        function [obs, reward, done, logged] = step(obj, action)
            [reward, done] = obj.BaseOptimizer.update(action);
            obs = obj.getObservation();
        end
        
        function obs = getObservation(obj)
            % 构造观测向量[N x (D+M+1)]
            pop = obj.BaseOptimizer.Population.decs;
            objs = obj.BaseOptimizer.Population.objs;
            cvs = sum(max(0,obj.BaseOptimizer.Population.cons),2);
            obs = [pop, objs, cvs];
            obs = dlarray(obs,'TC');
        end
        
        function InitialObservation = reset(obj)
            % 重置环境
            obj.Problem = obj.ProblemSet{randi(numel(obj.ProblemSet))};
            obj.BaseOptimizer.Init(obj.Problem);
            InitialObservation = obj.getObservation();
        end
    end
end
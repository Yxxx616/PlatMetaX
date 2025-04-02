classdef Transformer_DE_UpEq_Environment < rl.env.MATLABEnvironment
    properties
        BaseOptimizer    % 基础优化器实例
        Problem          % cur problem
        ProblemSet       % ProblemSet
        StateEncoder     % 状态编码器
        RewardWeights = [0.7, 0.3] % 可行率与最优值权重
    end
    
    methods
        function obj = Transformer_DE_UpEq_Environment(ps, bo, task)
            % 初始化观测和动作空间
            maxTreeDepth = 3;
            problem = ps{1};
            symbolBits = 4; % 每个符号4位编码
            treeNodes = 2^(maxTreeDepth+1)-1; % 完全二叉树节点数
            actionSize = symbolBits * treeNodes;
            ObservationInfo = rlNumericSpec([problem.N, problem.D+2]);
            ActionInfo = rlNumericSpec([symbolBits treeNodes],...
                'LowerLimit',0,'UpperLimit',1);
            % 调用父类构造函数
            obj = obj@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            % 创建基础优化器
            obj.BaseOptimizer = bo();
            obj.Problem = problem;
            obj.ProblemSet = ps;
        end
        
        function [obs, reward, done, logged] = step(obj, action)
            obj.BaseOptimizer.update(action);
            
            % 计算状态变化
            prevFeasible = obj.calculateFeasibleRate();
            prevBest = obj.BaseOptimizer.BestSolution.obj;
            
            % 更新种群
            [newBest, ~] = obj.BaseOptimizer.findBestIndividual();
            
            % 计算奖励
            newFeasible = obj.calculateFeasibleRate();
            deltaFeasible = newFeasible - prevFeasible;
            deltaBest = (prevBest - newBest.obj)/abs(prevBest);
            
            reward = obj.RewardWeights(1)*deltaFeasible + ...
                     obj.RewardWeights(2)*deltaBest;
            
            % 生成新观测
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
    
    methods (Access = private)
        function feasibleRate = calculateFeasibleRate(obj)
            % 计算可行率
            cvs = sum(max(0,obj.BaseOptimizer.Population.cons),2);
            feasibleRate = sum(cvs <= 0) / numel(cvs);
        end
        
    end
end
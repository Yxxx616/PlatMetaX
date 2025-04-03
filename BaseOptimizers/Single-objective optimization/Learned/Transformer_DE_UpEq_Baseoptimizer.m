classdef Transformer_DE_UpEq_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none> <learned/none>

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: A MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    properties
        Population      % 当前种群
        BestSolution    % 历史最佳解
        CR              % 交叉概率
        F               % 缩放因子
        StrategyParams  % 变异策略参数
        Problem         % 当前优化问题
        MaxTreeDepth=3    % 最大表达式树深度
        SymbolDict = containers.Map(...
            {'+', '-', '×', 'x', 'x*', 'xr', 'c', 'Δx', 'xi*'},...
            {'0001','0011','0010','0101','0110','1001','0100','1000','1010'});
        InvSymbolDict = containers.Map(...
            {'0001','0011','0010','0101','0110','1001','0100','1000','1010'},...
            {'+', '-', '×', 'x', 'x*', 'xr', 'c', 'Δx', 'xi*'});
    end
    
    methods        
        function init(obj, problem)
            % 初始化优化器
            obj.Problem = problem;
            obj.Population = problem.Initialization();
            [obj.BestSolution, ~] = obj.findBestIndividual();
        end
        
        function [newPop, done] = update(this, action)
            % 执行一代进化
            % 解码符号树
            expr = baseOptimizer.decodeTree(action);
            
            % 执行变异操作
            offspring = this.applyMutation(expr, this.Population.decs);
            Offspring = this.pro.Evaluation(offspring);
            replace             = FitnessSingle(this.Population) > FitnessSingle(Offspring);
            this.Population(replace) = Offspring(replace);
            % 环境选择
            newPop = this.Population;
            done = ~this.NotTerminated(this.Population);
        end
        
        function strategyExpr = decodeTree(obj, binaryCode)
            % 将二进制编码转换为符号表达式树
            bitStream = char(binaryCode(:)' + '0');
            symbols = {};
            
            % 每4位分割解析符号
            for i = 1:4:length(bitStream)
                code = bitStream(i:min(i+3,end));
                if obj.InvSymbolDict.isKey(code)
                    symbols{end+1} = obj.InvSymbolDict(code);
                else
                    symbols{end+1} = 'c'; % 无法识别则视为常数
                end
            end
            
            % 构建表达式树（示例实现前序遍历构建）
            [strategyExpr, ~] = obj.buildExpressionTree(symbols, 1);
        end
        
        function [expr, idx] = buildExpressionTree(obj, symbols, startIdx)
            % 递归构建表达式树
            if startIdx > length(symbols)
                expr = [];
                idx = startIdx;
                return;
            end
            
            token = symbols{startIdx};
            expr = struct('op', token, 'children', []);
            idx = startIdx + 1;
            
            % 根据运算符确定子节点数
            if ismember(token, {'+', '-', '×'})
                [left, idx] = obj.buildExpressionTree(symbols, idx);
                [right, idx] = obj.buildExpressionTree(symbols, idx);
                expr.children = {left, right};
            elseif strcmp(token, 'c')
                % 常数节点需要解析数值
                expr.value = obj.parseConstant(symbols, idx);
                idx = idx + 2; % 跳过mantissa和exponent位
            end
        end
        
        function value = parseConstant(~, symbols, idx)
            % 解析常数c的mantissa和exponent（简化实现）
            if idx+1 <= length(symbols)
                mantissa = bin2dec(symbols{idx})/10;
                exponent = bin2dec(symbols{idx+1}) - 2; % 允许指数为-1,0
                value = mantissa * 10^exponent;
            else
                value = 0.5; % 默认值
            end
        end
        
        function newPop = applyMutation(obj, strategyExpr, pop)
            % 执行符号表达式指导的变异
            N = size(pop, 1);
            newPop = zeros(size(pop));
            
            for i = 1:N
                % 获取当前个体上下文
                context.x = pop(i,:);
                context.x_best = obj.BestSolution;
                context.x_rand = pop(randi(N),:);
                
                % 递归计算表达式值
                delta = obj.evaluateExpr(strategyExpr, context);
                newPop(i,:) = pop(i,:) + delta;
            end
        end
        
        function val = evaluateExpr(obj, expr, context)
            % 递归计算表达式值
            switch expr.op
                case '+'
                    val = obj.evaluateExpr(expr.children{1}, context) + ...
                           obj.evaluateExpr(expr.children{2}, context);
                case '-'
                    val = obj.evaluateExpr(expr.children{1}, context) - ...
                           obj.evaluateExpr(expr.children{2}, context);
                case '×'
                    val = obj.evaluateExpr(expr.children{1}, context) .* ...
                           obj.evaluateExpr(expr.children{2}, context);
                case 'x'
                    val = context.x;
                case 'x*'
                    val = context.x_best;
                case 'xr'
                    val = context.x_rand;
                case 'Δx'
                    val = context.x - context.x_prev;
                case 'c'
                    val = expr.value;
                otherwise
                    val = 0;
            end
        end
        function [best, idx] = findBestIndividual(this)
            % 找到当前最优解
            [~, idx] = min(this.Population.objs);
            best = this.Population(idx);
        end
    end
end


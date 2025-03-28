function state = calSOPState(bo)
%计算无约束单目标优化特征
    % 获取决策变量矩阵和目标值矩阵
    %decs 是一个 N x D 矩阵，包含解决方案
    % objs 是一个 N x 1 矩阵，包含适应度值
    decs = bo.Population.decs;
    objs = bo.Population.objs;
    n = size(decs, 1); % Number of individuals
    
    state = zeros(13,1);
    state(1) = min(objs);
    state(2) = max(objs);
    state(3) = mean(objs);
    state(4) = std(objs);

    dist_matrix = squareform(pdist(decs));
    state(5) = sum(mean(dist_matrix(:)));
    
    state(6) = sum(std(decs));
    
%% Fitness distance correlation
    % 计算最优解（最优解是适应度值最小的解）
    [~, best_idx] = min(objs);
    best_solution = decs(best_idx, :);

    % 计算每个解到最优解的欧几里得距离
    D = zeros(n, 1);
    for i = 1:n
        D(i) = norm(decs(i, :) - best_solution);
    end

    % 计算适应度值的平均值和方差
    f_bar = state(3);
    delta_F = state(4);

    % 计算距离的平均值和方差
    d_bar = mean(D);
    delta_D = std(D);

    % 计算适应度值和距离的协方差
    C_FD = 0;
    for i = 1:n
        C_FD = C_FD + (objs(i) - f_bar) * (D(i) - d_bar);
    end
    C_FD = C_FD / n;

    % 计算 FDC
    state(7) = C_FD / (delta_F * delta_D);
    
%%  最优值个数
    % 根据距离对个体进行排序
    [~, idx] = sort(D);
    sorted_objs = objs(idx);

    % 初始化局部适应度景观评估指标
    chi = 0;

    % 计算局部适应度景观评估指标
    for i = 1:n-1
        if sorted_objs(i+1) <= sorted_objs(i)
            chi = chi + 1;
        end
    end

    % 归一化指标
    state(8)  = chi / n;
    
%% QUANTILES

    p = 0:0.25:1;
    y = quantile(objs,p);
    state(9:end) = y';
end
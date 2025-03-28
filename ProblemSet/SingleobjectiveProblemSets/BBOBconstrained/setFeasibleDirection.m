function feasibleDirection = setFeasibleDirection(feasibleDirection, xopt, dimension, rseed)
% 常量定义
    seed_offset = 412;  % 在 0-999 之间均匀采样
    feas_shrink = 0.75;  % 在 0.75 和 1.0 之间随机缩放
    feas_bound = 5.0;    % 可行边界
    
    % 初始化最大绝对值和最大相对值
    maxabs = 0;
    maxrel = 0;
    
    % 计算 maxabs 和 maxrel
    for i = 1:dimension
        maxabs = max(maxabs, abs(xopt(i)));
        maxrel = max(maxrel, feasibleDirection(i) / (feas_bound - xopt(i)));
        maxrel = max(maxrel, feasibleDirection(i) / (-feas_bound - xopt(i)));
    end
    
    % 检查 maxabs 是否超过 4.01
    if maxabs > 4.01
        warning('feasible_direction_set_length: a component of fabs(xopt) was greater than 4.01');
    end
    
    % 检查 maxabs 是否超过 5.0
    if maxabs > 5.0
        error('feasible_direction_set_length: a component of fabs(xopt) was greater than 5.0');
    end
    
    % 生成随机数
    rng(rseed + seed_offset);
    r = rand(1);
    
    % 缩放 feasible_direction
    feasibleDirection = feasibleDirection * (feas_shrink + r * (1 - feas_shrink)) / maxrel;
end


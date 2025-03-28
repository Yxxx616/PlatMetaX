function state = calSCOPState(bo)
%计算约束单目标优化特征
    % 识别可行点
    state = zeros(14,1);
    consSum = sum(bo.Population.cons,2);
    objVec = bo.Population.objs;
    feasible_mask = any(bo.Population.cons<=0,2);
    if sum(feasible_mask)> 0
        feasible_points = bo.Population(feasible_mask).decs;
        % 使用DBSCAN聚类算法识别可行组件
        epsilon = 0.5; % 邻域半径，需要根据数据调整
        minpts = 10; % 邻域内最小点数，需要根据数据调整
        [idx, ~] = dbscan(feasible_points, epsilon, minpts);

        % 计算可行组件数量
        NF = numel(unique(idx)) - 1; % 减1是因为0是为噪声点分配的
        state(1) = NF/ sum(feasible_mask);
        % 计算可行性比率
        qF = sum(feasible_mask) / bo.NP;
        state(2) = qF;

        % 计算可行边界交叉比率
        % 计算边界交叉数量
        crossings = sum(diff([0, feasible_mask', 0]) ~= 0);
        RFBx = crossings / (bo.NP - 1);
        state(3) = RFBx;
    end

    FVC = corr(objVec, consSum, 'Type', 'Spearman');
    state(4) = FVC;
    minFitness = min(objVec, [], 1);
    maxFitness = max(objVec, [], 1);

    % 计算约束违反程度的最小值和最大值
    minViolation = min(consSum);
    maxViolation = max(consSum);
    state(5) = minFitness / (maxFitness+1e-6);
    state(6) = minViolation / (maxViolation+1e-6);

    % 确定理想区域的范围
    % 对于每个目标，理想区域的范围是目标最小值加上25%的目标范围
    % 对于约束违反程度，理想区域的范围是约束最小值加上25%的约束范围
    idealZoneFitness = minFitness + 0.25 * (maxFitness - minFitness);
    idealZoneViolation = minViolation + 0.25 * (maxViolation - minViolation);

    % 计算在理想区域内的点的数量
    inIdealZone = sum(all(objVec <= idealZoneFitness, 2) & consSum <= idealZoneViolation);

    % 计算理想区域比例
    PiIZ0_25 = inIdealZone / bo.NP;

    % 对于 1% 的理想区域，可以类似地计算
    idealZoneFitness = minFitness + 0.01 * (maxFitness - minFitness);
    idealZoneViolation = minViolation + 0.01 * (maxViolation - minViolation);
    inIdealZone = sum(all(objVec <= idealZoneFitness, 2) & consSum <= idealZoneViolation);
    PiIZ0_01 = inIdealZone / bo.NP;

    state(7) = PiIZ0_25;
    state(8) = PiIZ0_01;
    state(9) = sum(consSum);
    state(10) = minFitness;
    state(11) = maxFitness;
    state(12) = minViolation;
    state(13) = maxViolation;
    state(14) = mean(consSum);
end
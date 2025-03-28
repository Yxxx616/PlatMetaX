function features = calFeatures(bo)
    %计算无约束单目标优化特征
    % 获取决策变量矩阵和目标值矩阵
    %decs 是一个 N x D 矩阵，包含解决方案
    % objs 是一个 N x 1 矩阵，包含适应度值
    decs = bo.Population.decs;
    objs = bo.Population.objs;
    n = size(decs, 1); % Number of individuals
    
    features = [min(objs); max(objs); mean(objs); std(objs); calculate_ela_distribution(objs);calculate_ela_meta(decs,objs)];
end


function metaF = calculate_ela_meta(X, y)
    % 计算线性模型的特征
    % 输入: X (设计矩阵), y (目标值)
    % 输出: features (包含特征的结构体)

    % 线性回归模型
    mdl = fitlm(X, y);
    
    % 提取特征
    metaF = [mdl.Coefficients.Estimate(1);mdl.Coefficients.Estimate(2:end);mdl.Rsquared.Adjusted;]; %截距 %系数 %调整后的 R²
end

function pcaF = calculate_pca(X, y)
    % 计算 PCA 特征
    % 输入: X (设计矩阵), y (目标值)
    % 输出: features (包含特征的结构体)

    % 合并 X 和 y
    data = [X, y];
    
    % 计算 PCA
    [coeff, ~, latent] = pca(data);
    
    % 提取特征  
    pcaF = [latent ./ sum(latent); latent(1) / sum(latent)]; % 解释方差  % 第一主成分的解释方差
end

function nbcF = calculate_nbc(X, y)
    % 计算最近邻特征
    % 输入: X (设计矩阵), y (目标值)
    % 输出: features (包含特征的结构体)

    % 计算最近邻距离
    nbrs = createns(X, 'NSMethod', 'kdtree');
    [~, dist] = knnsearch(nbrs, X, 'K', 2); % 找到每个点的最近邻
    
    % 提取特征
    nbcF = zeros(2,1); 
    nbcF(1) = mean(dist(:, 2)); % 最近邻距离的均值
    nbcF(2) = std(dist(:, 2)); % 最近邻距离的标准差
end

function difF = calculate_ela_distribution(y)
    % 计算目标值的分布特征
    % 输入: y (目标值)
    % 输出: 峰值数量

    % 计算偏度和峰度
    ske = skewness(y); % 偏度
    kur = kurtosis(y); % 峰度
    
    % 计算峰值数量
    [f, xi] = ksdensity(y); % 核密度估计
    peaks = findpeaks(f); % 找到峰值
    difF = [ske;kur;length(peaks)]; % 峰值数量
end
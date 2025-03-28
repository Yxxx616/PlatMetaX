% MATLAB Code
function [offspring] = updateFunc1125(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Population analysis
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c = max(0, cons);
    c_max = max(c) + eps;
    
    % 2. Adaptive weights
    w_f = exp(-0.5*((popfits-f_mean)./f_std).^2);
    w_c = 1 - tanh(c/c_max);
    w = 0.6*w_f + 0.4*w_c;
    w = w(:);
    
    % 3. Direction vectors
    elite_mask = w > 0.8;
    if any(elite_mask)
        elite_w = w(elite_mask);
        d_elite = sum((popdecs(elite_mask,:) - x_best) .* elite_w', 1) / (sum(elite_w) + eps);
    else
        d_elite = zeros(1,D);
    end
    
    div_mask = w < 0.3;
    if any(div_mask)
        d_div = sum(x_worst - popdecs(div_mask,:), 1) / (sum(div_mask) + eps);
    else
        d_div = zeros(1,D);
    end
    
    % 4. Mutation
    F1 = 0.9; F2 = 0.5; F3 = 0.7;
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_diff = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F1.*w.*d_elite + F2.*(1-w).*d_div + F3.*d_diff;
    
    % 5. Crossover
    CR = 0.5 + 0.4*(1-w);
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling
    out_lb = offspring < lb;
    out_ub = offspring > ub;
    offspring(out_lb) = lb(out_lb) + rand(sum(out_lb(:)),1).*(x_best(out_lb) - lb(out_lb));
    offspring(out_ub) = ub(out_ub) - rand(sum(out_ub(:)),1).*(ub(out_ub) - x_best(out_ub));
end
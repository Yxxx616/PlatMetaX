% MATLAB Code
function [offspring] = updateFunc1126(popdecs, popfits, cons)
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
    norm_f = (popfits - f_mean) ./ f_std;
    w_f = exp(-norm_f.^2);
    w_c = 1 - tanh(c/c_max);
    w = 0.7*w_f + 0.3*w_c;
    w = w(:);
    
    % 3. Direction vectors
    elite_mask = w > 0.5;
    if any(elite_mask)
        elite_w = w(elite_mask);
        d_elite = sum((popdecs(elite_mask,:) - x_best) .* elite_w, 1) / (sum(elite_w) + eps);
    else
        d_elite = zeros(1,D);
    end
    
    div_mask = w < 0.2;
    if any(div_mask)
        d_div = sum(x_worst - popdecs(div_mask,:), 1) / (sum(div_mask) + eps);
    else
        d_div = zeros(1,D);
    end
    
    % 4. Mutation
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    F = 0.5 + 0.4*(1-w);
    v_base = x_best + w.*d_elite + (1-w).*d_div;
    mutants = v_base + F.*(popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Crossover
    CR = 0.6 + 0.3*(1-w);
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection
    out_lb = offspring < lb;
    out_ub = offspring > ub;
    rnd = rand(NP,D);
    offspring(out_lb) = lb(out_lb) + rnd(out_lb).*(x_best(out_lb) - lb(out_lb));
    offspring(out_ub) = ub(out_ub) - rnd(out_ub).*(ub(out_ub) - x_best(out_ub));
end
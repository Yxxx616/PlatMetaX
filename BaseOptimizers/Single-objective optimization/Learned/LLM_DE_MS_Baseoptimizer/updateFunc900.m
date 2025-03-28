% MATLAB Code
function [offspring] = updateFunc900(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate fitness statistics and weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    weights = 1 ./ (1 + exp(5*(popfits - mean_fit)/std_fit));
    weights = weights / sum(weights);
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r = candidates(randperm(length(candidates), 4));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3); r4(i) = r(4);
    end
    
    % 3. Compute fitness-guided direction
    diff_f = bsxfun(@minus, popdecs, popdecs);
    weighted_diff = bsxfun(@times, diff_f, weights);
    sum_diff = sum(weighted_diff, 1);
    norm_sum = sqrt(sum(sum_diff.^2)) + eps;
    d_f = bsxfun(@rdivide, sum_diff, norm_sum);
    
    % 4. Compute constraint-aware direction
    max_cons = max(abs(cons)) + eps;
    c_weights = tanh(abs(cons)/max_cons) .* sign(cons);
    diff_c = popdecs(r1,:) - popdecs(r2,:);
    norm_c = sqrt(sum(diff_c.^2, 2)) + eps;
    d_c = bsxfun(@times, c_weights, diff_c ./ norm_c(:,ones(1,D)));
    
    % 5. Compute random diversity direction
    d_r = popdecs(r3,:) - popdecs(r4,:);
    
    % 6. Adaptive balance factor
    alpha = 1 ./ (1 + exp(-5*(popfits - mean_fit)/std_fit));
    
    % 7. Composite mutation
    F = 0.5 + 0.3 * rand(NP,1);
    term1 = bsxfun(@times, alpha, d_f);
    term2 = bsxfun(@times, (1-alpha), d_c);
    mutants = popdecs + bsxfun(@times, F, term1 + term2) + 0.5 * bsxfun(@times, F, d_r);
    
    % 8. Adaptive crossover
    min_fit = min(popfits);
    max_fit = max(popfits);
    CR = 0.2 + 0.6 * (popfits - min_fit) / (max_fit - min_fit + eps);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 9. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 10. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    norm_fit = (popfits - min_fit) / (max_fit - min_fit + eps);
    reflect_factor = 0.1 * (1 - norm_fit);
    reflect_factor = repmat(reflect_factor, 1, D);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (ub - lb);
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (ub - lb);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end
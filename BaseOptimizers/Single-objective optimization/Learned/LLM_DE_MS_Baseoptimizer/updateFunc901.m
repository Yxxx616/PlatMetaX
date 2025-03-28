% MATLAB Code
function [offspring] = updateFunc901(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate fitness weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    weights = 1 ./ (1 + exp(5*(popfits - mean_fit)/std_fit));
    weights = weights / sum(weights);
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r = candidates(randperm(length(candidates), 3));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3);
    end
    
    % 3. Fitness-guided direction
    pop_rep = reshape(popdecs, [1 NP D]);
    diff_f = bsxfun(@minus, popdecs, pop_rep);
    weighted_diff = bsxfun(@times, diff_f, reshape(weights, [1 NP 1]));
    sum_diff = sum(weighted_diff, 2);
    norm_sum = sqrt(sum(sum_diff.^2, 3)) + eps;
    d_f = bsxfun(@rdivide, sum_diff, norm_sum);
    d_f = squeeze(d_f);
    
    % 4. Constraint-aware direction
    max_cons = max(abs(cons)) + eps;
    c_weights = tanh(abs(cons)/max_cons) .* sign(cons);
    diff_c = popdecs(r1,:) - popdecs(r2,:);
    norm_c = sqrt(sum(diff_c.^2, 2)) + eps;
    d_c = bsxfun(@times, c_weights, diff_c ./ norm_c(:,ones(1,D)));
    
    % 5. Diversity direction
    d_r = popdecs(r1,:) - popdecs(r3,:);
    
    % 6. Adaptive balance factor
    alpha = 1 ./ (1 + exp(-5*(popfits - mean_fit)/std_fit));
    
    % 7. Composite mutation
    F = 0.5 + 0.3 * rand(NP,1);
    term1 = bsxfun(@times, alpha, d_f);
    term2 = bsxfun(@times, (1-alpha), d_c);
    mutants = popdecs + bsxfun(@times, F, term1 + term2) + 0.3 * bsxfun(@times, F, d_r);
    
    % 8. Adaptive crossover
    min_fit = min(popfits);
    max_fit = max(popfits);
    CR = 0.1 + 0.7 * (popfits - min_fit) / (max_fit - min_fit + eps);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 9. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 10. Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
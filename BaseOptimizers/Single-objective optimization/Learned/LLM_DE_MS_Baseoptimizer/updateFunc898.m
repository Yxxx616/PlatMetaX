% MATLAB Code
function [offspring] = updateFunc898(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute fitness-weighted centroid
    mean_fit = mean(popfits);
    weights = 1 ./ (1 + exp(popfits - mean_fit));
    weights = weights / sum(weights);
    centroid = sum(bsxfun(@times, popdecs, weights), 1);
    
    % 2. Constraint-aware parameters
    max_cons = max(abs(cons)) + eps;
    alpha = 1 - tanh(abs(cons)/max_cons);
    beta = 0.5 * (1 - alpha);
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r = candidates(randperm(length(candidates), 4));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3); r4(i) = r(4);
    end
    
    % 4. Directional perturbation
    diff = popdecs(r1,:) - popdecs(r2,:);
    norm_diff = sqrt(sum(diff.^2, 2)) + eps;
    d = bsxfun(@times, sign(cons), diff ./ norm_diff(:,ones(1,D)));
    
    % 5. Composite mutation
    F = 0.6;
    term1 = bsxfun(@times, alpha, centroid - popdecs);
    term2 = bsxfun(@times, beta, d);
    term3 = bsxfun(@times, (1-alpha), popdecs(r3,:) - popdecs(r4,:));
    mutants = popdecs + F * (term1 + term2) + term3;
    
    % 6. Adaptive crossover
    min_fit = min(popfits);
    max_fit = max(popfits);
    CR = 0.3 + 0.5 * (popfits - min_fit) / (max_fit - min_fit + eps);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    norm_fit = (popfits - min_fit) / (max_fit - min_fit + eps);
    reflect_factor = 0.1 + 0.4 * (1 - norm_fit);
    reflect_factor = repmat(reflect_factor, 1, D);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (lb_rep(below) - offspring(below));
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (offspring(above) - ub_rep(above));
    
    % 9. Final clamping with small perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    offspring = offspring + 0.01 * (ub - lb) .* randn(NP,D) .* (1 - norm_fit(:,ones(1,D)));
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % 10. Elite preservation (top 5%)
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, round(0.05*NP));
    elite = popdecs(sorted_idx(1:elite_size),:);
    offspring(sorted_idx(1:elite_size),:) = elite + 0.002 * (ub - lb) .* randn(elite_size, D);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
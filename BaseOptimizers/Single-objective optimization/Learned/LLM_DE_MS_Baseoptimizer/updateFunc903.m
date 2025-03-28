% MATLAB Code
function [offspring] = updateFunc903(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate fitness weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    range_fit = max_fit - min_fit + eps;
    weights = exp(-(popfits - min_fit)/range_fit);
    
    % 2. Compute weighted centroid
    weighted_sum = sum(bsxfun(@times, popdecs, weights), 1);
    centroid = weighted_sum / sum(weights);
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r = candidates(randperm(length(candidates), 4));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3); r4(i) = r(4);
    end
    
    % 4. Constraint-aware direction
    max_cons = max(abs(cons)) + eps;
    c_weights = tanh(abs(cons)/max_cons) .* sign(cons);
    diff_c = popdecs(r1,:) - popdecs(r2,:);
    d_c = bsxfun(@times, c_weights, diff_c);
    
    % 5. Adaptive scaling factors
    F = 0.4 + 0.4 * (popfits - min_fit)/range_fit;
    
    % 6. Composite mutation
    centroid_diff = bsxfun(@minus, centroid, popdecs);
    rand_diff = popdecs(r3,:) - popdecs(r4,:);
    mutants = popdecs + bsxfun(@times, F, centroid_diff) + ...
              bsxfun(@times, (1-F), d_c) + 0.1 * rand_diff;
    
    % 7. Rank-based crossover
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.1 + 0.8 * ranks/NP;
    
    % 8. Crossover
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
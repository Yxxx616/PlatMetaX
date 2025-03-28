% MATLAB Code
function [offspring] = updateFunc564(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, min_cons_idx] = min(cons);
        elite = popdecs(min_cons_idx, :);
    end
    
    % Calculate ranks for fitness and constraints
    [~, rank_fit] = sort(popfits);
    [~, rank_cons] = sort(cons);
    rank_fit = rank_fit / NP;
    rank_cons = rank_cons / NP;
    
    % Normalize constraint violations
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Generate four distinct random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r3 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r4 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Calculate scaling factors
    F1 = 0.8 * (1 - rank_fit) .* (1 - norm_cons);
    F2 = 0.4 * rank_fit .* norm_cons;
    
    % Calculate mutation components
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    
    % Adaptive perturbation
    sigma = 0.1 * (1 + rank_cons);
    perturbation = bsxfun(@times, sigma, randn(NP, D));
    
    % Combined mutation
    offspring = popdecs + ...
        bsxfun(@times, F1, elite_diff) + ...
        bsxfun(@times, F2, rand_diff) + ...
        perturbation;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low + ...
        (2*ub_rep - offspring) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
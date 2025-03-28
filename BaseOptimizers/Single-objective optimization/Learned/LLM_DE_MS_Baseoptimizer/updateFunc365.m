% MATLAB Code
function [offspring] = updateFunc365(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint processing
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    max_cons = max(abs_cons) + eps;
    norm_cons = abs_cons / max_cons;
    
    % Elite selection
    [~, elite_idx] = min(popfits);
    elite = popdecs(elite_idx, :);
    
    % Feasible leader selection
    if any(feasible_mask)
        feasible_fits = popfits(feasible_mask);
        [~, best_feas_idx] = min(feasible_fits);
        temp = find(feasible_mask);
        leader = popdecs(temp(best_feas_idx), :);
    else
        [~, best_cons_idx] = min(abs_cons);
        leader = popdecs(best_cons_idx, :);
    end
    
    % Adaptive parameters
    alpha = 0.7 - 0.3 * norm_cons;
    beta = 0.4 + 0.3 * (1 - norm_cons);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Ensure unique indices
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    
    % Differential vectors
    d_explore = popdecs(r1,:) - popdecs(r2,:);
    d_exploit = bsxfun(@minus, leader, popdecs);
    
    % Mutation core
    scaled_fits = 0.5 + 0.5 * (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    offspring = popdecs + bsxfun(@times, alpha .* beta, d_exploit) + ...
        bsxfun(@times, (1 - alpha) .* scaled_fits, d_explore);
    
    % Constraint-driven perturbation
    sigma = 0.2 * (1 + norm_cons);
    offspring = offspring + bsxfun(@times, sigma, randn(NP, D));
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring .* (~out_low & ~out_high) + ...
        (2 * lb - offspring) .* out_low + ...
        (2 * ub - offspring) .* out_high;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end
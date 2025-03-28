% MATLAB Code
function [offspring] = updateFunc364(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % Constraint handling
    phi = max(0, cons);
    feasible_mask = phi <= eps_val;
    max_phi = max(phi) + eps_val;
    
    % Find best individual
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        x_best = popdecs(temp(best_idx),:);
    else
        [~, best_idx] = min(phi);
        x_best = popdecs(best_idx,:);
    end
    
    % Adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.5 * (1 + tanh(1 - (popfits - f_min)/(f_max - f_min + eps_val)));
    w = phi / max_phi;
    
    % Generate random indices
    r1 = mod((1:NP)' + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod((1:NP)' + randi(NP-1, NP, 1), NP) + 1;
    
    % Mutation vectors
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    direction_vectors = bsxfun(@minus, x_best, popdecs);
    mutation_terms = bsxfun(@times, (1-w), diff_vectors) + bsxfun(@times, w, direction_vectors);
    v = popdecs + bsxfun(@times, F, mutation_terms);
    
    % Constraint-aware perturbation
    sigma = 0.2 * (1 - exp(-5*w));
    offspring = v + bsxfun(@times, sigma, randn(NP,D));
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring.*(~out_low & ~out_high) + ...
               (2*lb - offspring).*out_low + ...
               (2*ub - offspring).*out_high;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end
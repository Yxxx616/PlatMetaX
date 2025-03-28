% MATLAB Code
function [offspring] = updateFunc360(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint handling
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    
    % Elite selection (best overall fitness)
    [~, elite_idx] = min(popfits);
    elite = popdecs(elite_idx,:);
    
    % Best feasible individual
    if any(feasible_mask)
        feasible_fits = popfits(feasible_mask);
        [~, best_feas_idx] = min(feasible_fits);
        temp = find(feasible_mask);
        best_feas = popdecs(temp(best_feas_idx),:);
    else
        [~, best_cons_idx] = min(abs_cons);
        best_feas = popdecs(best_cons_idx,:);
    end
    
    % Generate unique random pairs avoiding current index
    r1 = arrayfun(@(x) randi([1, NP-1]), 1:NP)';
    r1 = r1 + (r1 >= (1:NP)');
    r2 = arrayfun(@(x) randi([1, NP-1]), 1:NP)';
    r2 = r2 + (r2 >= (1:NP)');
    
    % Rank-based scaling factors (cosine variation)
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    alpha = 0.5 + 0.3 * cos(pi * ranks/NP);
    
    % Constraint-based weights
    max_cons = max(abs_cons) + eps;
    beta = 0.2 + 0.6 * (1 - exp(-5 * abs_cons / max_cons));
    
    % Direction vectors
    d_feas = bsxfun(@minus, best_feas, popdecs);
    d_elite = bsxfun(@minus, elite, popdecs);
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation with adaptive weights
    offspring = popdecs + bsxfun(@times, alpha, beta.*d_feas + (1-beta).*d_elite) + ...
                bsxfun(@times, (1-alpha), d_rand);
    
    % Constraint-driven perturbation
    perturbation = randn(NP,D) .* (1 - exp(-abs_cons/max_cons));
    offspring = offspring + bsxfun(@times, perturbation, (ub-lb)*0.1);
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + abs(offspring - lb)).*out_low + ...
               (ub - abs(offspring - ub)).*out_high;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end
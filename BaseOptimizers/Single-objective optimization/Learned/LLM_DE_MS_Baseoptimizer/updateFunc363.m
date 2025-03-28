% MATLAB Code
function [offspring] = updateFunc363(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint handling
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    max_cons = max(abs_cons) + eps;
    
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
    
    % Generate unique random pairs
    r1 = mod((1:NP)' + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod((1:NP)' + randi(NP-1, NP, 1), NP) + 1;
    
    % Rank-based scaling factors
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    alpha = 0.6 + 0.2 * cos(pi * ranks/NP);
    
    % Constraint-based weights (sigmoid transformation)
    beta = 0.5 * (1 + tanh(5 * (1 - abs_cons/max_cons)));
    
    % Elite guidance vectors
    F = 0.5 + 0.3 * randn(NP,1);
    v_elite = elite + F .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Feasible direction vectors
    d_feas = bsxfun(@minus, best_feas, popdecs);
    
    % Combined mutation
    v = popdecs + bsxfun(@times, alpha, beta.*d_feas + (1-beta).*v_elite);
    
    % Constraint-driven perturbation
    sigma = 0.2 * (ub - lb) .* (1 - exp(-abs_cons/max_cons));
    offspring = v + randn(NP,D) .* sigma;
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring.*(~out_low & ~out_high) + ...
               (2*lb - offspring).*out_low + ...
               (2*ub - offspring).*out_high;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end
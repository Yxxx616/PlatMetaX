% MATLAB Code
function [offspring] = updateFunc625(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Adaptive scaling factors incorporating both fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons)) + eps;
    
    F = 0.5 * (1 + (cons./c_max) .* ((popfits - f_min)./(f_max - f_min + eps)));
    F = min(max(F, 0.1), 0.9);  % Clamp to [0.1, 0.9]
    
    % 3. Constraint-directed mutation
    alpha = 0.1;
    perturbation = alpha * cons .* randn(NP, D);
    
    % Random indices selection (vectorized)
    r1 = randi(NP-1, NP, 1);
    r2 = randi(NP-2, NP, 1);
    r1 = r1 + (r1 >= (1:NP)');
    r2 = r2 + (r2 >= min(r1, (1:NP)'));
    r2 = r2 + (r2 >= max(r1, (1:NP)'));
    
    elite_rep = repmat(elite, NP, 1);
    diff = popdecs(r1,:) - popdecs(r2,:);
    F_rep = repmat(F, 1, D);
    mutant = elite_rep + F_rep.*diff + perturbation;
    
    % 4. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    rank_ratio = (rank_idx-1)/(NP-1);
    CR = 0.9 - 0.5 * rank_ratio;
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
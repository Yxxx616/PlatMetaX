% MATLAB Code
function [offspring] = updateFunc1103(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints for weights
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    weights = 0.7 * f_norm + 0.3 * (1 - c_norm);
    
    % 2. Create four pools: elite, medium, diverse, constrained
    [~, sorted_idx] = sort(weights);
    elite_size = max(2, floor(NP*0.3));
    diverse_size = max(2, floor(NP*0.3));
    
    elite_pool = sorted_idx(end-elite_size+1:end);
    medium_pool = sorted_idx(elite_size+1:end-diverse_size);
    diverse_pool = sorted_idx(1:diverse_size);
    
    % Constraint-based pool (least violated)
    [~, cons_sorted] = sort(abs(cons));
    cons_pool = cons_sorted(1:elite_size);
    
    % 3. Generate indices for mutation
    idx_e1 = elite_pool(randi(elite_size, NP, 1));
    idx_e2 = elite_pool(randi(elite_size, NP, 1));
    idx_d1 = diverse_pool(randi(diverse_size, NP, 1));
    idx_d2 = diverse_pool(randi(diverse_size, NP, 1));
    idx_c1 = cons_pool(randi(elite_size, NP, 1));
    idx_c2 = cons_pool(randi(elite_size, NP, 1));
    
    % 4. Adaptive mutation factors
    F1 = 0.5 + 0.3 * weights;
    F2 = 0.8 - 0.5 * weights;
    Fc = 0.2 * (1 - c_norm);
    
    % 5. Mutation operation
    x_e1 = popdecs(idx_e1, :);
    x_e2 = popdecs(idx_e2, :);
    x_d1 = popdecs(idx_d1, :);
    x_d2 = popdecs(idx_d2, :);
    x_c1 = popdecs(idx_c1, :);
    x_c2 = popdecs(idx_c2, :);
    
    mutants = popdecs + F1.*(x_e1 - x_e2) + F2.*(x_d1 - x_d2) + Fc.*(x_c1 - x_c2);
    
    % 6. Directional crossover with adaptive rate
    CR = 0.85 - 0.5 * weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 0.5*(lb_rep(below) + popdecs(below));
    offspring(above) = 0.5*(ub_rep(above) + popdecs(above));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
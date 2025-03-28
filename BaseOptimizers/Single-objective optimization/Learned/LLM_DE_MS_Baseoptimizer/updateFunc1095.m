% MATLAB Code
function [offspring] = updateFunc1095(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate composite weights
    max_cons = max(abs(cons)) + eps;
    w_c = 1 - abs(cons)/max_cons;
    
    [~, ranks] = sort(popfits);
    w_f = ranks'/NP;
    weights = 0.7*w_c + 0.3*w_f;
    
    % 2. Create elite and diverse pools
    elite_size = max(3, ceil(NP*0.3));
    diverse_size = max(3, ceil(NP*0.3));
    [~, sorted_idx] = sort(weights, 'descend');
    elite_pool = sorted_idx(1:elite_size);
    diverse_pool = sorted_idx(end-diverse_size+1:end);
    
    % 3. Generate indices for mutation
    idx_e1 = elite_pool(randi(elite_size, NP, 1));
    idx_e2 = elite_pool(randi(elite_size, NP, 1));
    idx_d1 = diverse_pool(randi(diverse_size, NP, 1));
    idx_d2 = diverse_pool(randi(diverse_size, NP, 1));
    
    % 4. Adaptive mutation
    F = 0.5 + 0.3*weights;
    x_e1 = popdecs(idx_e1, :);
    x_e2 = popdecs(idx_e2, :);
    x_d1 = popdecs(idx_d1, :);
    x_d2 = popdecs(idx_d2, :);
    
    mutants = popdecs + F.*(x_e1 - x_e2) + (1-F).*(x_d1 - x_d2);
    
    % 5. Directed crossover
    CR = 0.9 - 0.4*weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end
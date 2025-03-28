% MATLAB Code
function [offspring] = updateFunc1078(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    
    % 2. Select elite pool (top 30%)
    [~, sorted_idx] = sort(popfits + 1000*norm_cons);
    elite_size = ceil(NP*0.3);
    elite_pool = sorted_idx(1:elite_size);
    
    % 3. Generate random indices for mutation
    idx1 = elite_pool(randi(elite_size, NP, 1));
    idx2 = elite_pool(randi(elite_size, NP, 1));
    idx3 = randperm(NP)';
    
    % 4. Calculate direction vectors
    x_best = popdecs(sorted_idx(1), :);
    delta_best = x_best(ones(NP,1), :) - popdecs;
    delta_elite1 = popdecs(idx1, :) - popdecs;
    delta_elite2 = popdecs(idx2, :) - popdecs;
    delta_rand = popdecs(idx3, :) - popdecs;
    
    % 5. Adaptive scaling factors
    F0 = 0.5;
    Ff = 0.3 * (1 - norm_fits);
    Fc = 0.2 * (1 - norm_cons);
    F = F0 + Ff + Fc;
    
    % 6. Mutation
    mutants = popdecs + F(:, ones(1,D)) .* (delta_best + delta_elite1 + delta_elite2 + delta_rand)/3;
    
    % 7. Constraint-aware crossover
    CR = 0.9 * (1 - norm_cons.^2);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % 10. Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end
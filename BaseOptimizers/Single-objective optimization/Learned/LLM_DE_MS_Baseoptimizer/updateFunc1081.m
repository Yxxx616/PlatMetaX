% MATLAB Code
function [offspring] = updateFunc1081(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % 2. Select elite pool (top 20%)
    combined_score = popfits + 100 * max(0, cons);
    [~, sorted_idx] = sort(combined_score);
    elite_size = ceil(NP*0.2);
    elite_pool = sorted_idx(1:elite_size);
    
    % 3. Generate random indices
    idx_best = sorted_idx(1);
    idx_elite1 = elite_pool(randi(elite_size, NP, 1));
    idx_elite2 = elite_pool(randi(elite_size, NP, 1));
    idx_rand1 = randi(NP, NP, 1);
    idx_rand2 = randi(NP, NP, 1);
    
    % 4. Calculate direction vectors
    x_best = popdecs(idx_best, :);
    delta_best = x_best(ones(NP,1), :) - popdecs;
    delta_elite = popdecs(idx_elite1, :) - popdecs(idx_elite2, :);
    delta_rand = popdecs(idx_rand1, :) - popdecs(idx_rand2, :);
    
    % 5. Adaptive scaling factors
    F0 = 0.5;
    Ff = 0.3 * (1 - norm_fits);
    Fc = 0.2 * exp(-norm_cons);
    F = F0 + Ff + Fc;
    
    % 6. Mutation
    mutants = popdecs + F(:, ones(1,D)) .* (delta_best + 0.5*delta_elite + 0.3*delta_rand);
    
    % 7. Constraint-aware crossover
    CR = 0.9 * (1 - tanh(norm_cons));
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    cons_rep = cons > 0;
    
    % For infeasible solutions
    below = offspring < lb_rep & cons_rep(:, ones(1,D));
    above = offspring > ub_rep & cons_rep(:, ones(1,D));
    offspring(below) = (lb_rep(below) + popdecs(below))/2;
    offspring(above) = (ub_rep(above) + popdecs(above))/2;
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end
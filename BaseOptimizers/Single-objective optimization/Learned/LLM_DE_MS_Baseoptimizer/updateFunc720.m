% MATLAB Code
function [offspring] = updateFunc720(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    iter = 720;
    max_iter = 1000;
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(abs(cons));
        elite = popdecs(elite_idx, :);
    end
    
    % Generate random indices (6 per individual)
    idx = arrayfun(@(x) randperm(NP, 6), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3);
    r4 = idx(:,4); r5 = idx(:,5); r6 = idx(:,6);
    
    % Adaptive scaling factor with sine modulation and constraint guidance
    F_base = 0.5 * (1 + sin(pi * iter / (2 * max_iter)));
    F = F_base .* (1 - w_c);
    
    % Create mutation vectors
    base = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    
    % Fitness-weighted difference
    diff_f = (popdecs(r3,:) - popdecs(r4,:)) .* w_f(:, ones(1, D));
    
    % Constraint-guided direction
    diff_c = (popdecs(r5,:) - popdecs(r6,:)) .* w_c(:, ones(1, D));
    
    % Combined mutation
    mutant = base + F .* diff1 + F .* diff_f + F .* diff_c;
    
    % Adaptive crossover rate
    CR = 0.9 * (1 - sqrt(w_f .* w_c));
    CR = min(max(CR, 0.1), 0.9);
    
    % Binomial crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection and perturbation
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    rand_perturb = 0.05 * (rand(NP, D) .* (ub_rep - lb_rep));
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb) + rand_perturb(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub) - rand_perturb(above_ub);
    
    % Final clipping
    offspring = max(min(offspring, ub_rep), lb_rep);
end
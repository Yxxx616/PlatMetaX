% MATLAB Code
function [offspring] = updateFunc723(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    iter = 723;
    max_iter = 1000;
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Elite selection with feasibility consideration
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, min_c_idx] = min(cons);
        [~, min_f_idx] = min(popfits);
        elite = (popdecs(min_c_idx,:) + popdecs(min_f_idx,:))/2;
    end
    
    % Generate random indices (6 per individual)
    idx = arrayfun(@(x) randperm(NP, 6), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3);
    r4 = idx(:,4); r5 = idx(:,5); r6 = idx(:,6);
    
    % Improved adaptive scaling factor
    F_base = 0.4 + 0.3 * cos(pi * iter / (2 * max_iter));
    F = F_base .* (1 - sqrt(w_c));
    
    % Enhanced mutation with directional components
    base = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff_f = (popdecs(r3,:) - popdecs(r4,:)) .* w_f(:, ones(1, D));
    diff_c = (popdecs(r5,:) - popdecs(r6,:)) .* (1-w_c(:, ones(1, D)));
    
    mutant = base + F .* (diff1 + diff_f + diff_c);
    
    % Dynamic crossover rate with wider range
    CR = 0.9 * (1 - sqrt(w_f .* w_c));
    CR = min(max(CR, 0.05), 0.95);
    
    % Binomial crossover with guaranteed change
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Improved boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    reflect_factor = 0.2 + 0.6*w_c(:, ones(1, D));
    offspring(below_lb) = lb_rep(below_lb) + reflect_factor(below_lb).*(lb_rep(below_lb) - offspring(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - reflect_factor(above_ub).*(offspring(above_ub) - ub_rep(above_ub));
    
    % Final clipping with reduced perturbation
    rand_perturb = 0.01 * (rand(NP, D) .* (ub_rep - lb_rep));
    offspring = offspring + rand_perturb;
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc725(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
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
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Adaptive scaling factors
    F_f = 0.4 + 0.3 * w_c;
    F_c = 0.4 + 0.3 * w_f;
    
    % Generate random indices (6 per individual)
    idx = arrayfun(@(x) randperm(NP, 6), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3);
    r4 = idx(:,4); r5 = idx(:,5); r6 = idx(:,6);
    
    % Enhanced directional mutation
    base = repmat(elite, NP, 1);
    diff_f = popdecs(r1,:) - popdecs(r2,:);
    diff_c = popdecs(r3,:) - popdecs(r4,:);
    diff_mix = popdecs(r5,:) - popdecs(r6,:);
    
    mutant = base + F_f .* diff_f + F_c .* diff_c + 0.5*(F_f + F_c) .* diff_mix;
    
    % Adaptive crossover rate
    CR = 0.85 - 0.25 * sqrt(w_f .* w_c);
    CR = max(min(CR, 0.9), 0.1);
    
    % Binomial crossover with guaranteed change
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection and perturbation
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    delta = 0.01 * (rand(NP,D)-0.5) .* (ub_rep - lb_rep);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb) + delta(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub) + delta(above_ub);
    
    % Final clipping with small perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    offspring = offspring + 0.005*(rand(NP,D)-0.5).*(ub_rep-lb_rep);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
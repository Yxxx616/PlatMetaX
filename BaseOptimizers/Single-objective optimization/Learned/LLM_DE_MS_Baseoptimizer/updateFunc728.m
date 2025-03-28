% MATLAB Code
function [offspring] = updateFunc728(popdecs, popfits, cons)
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
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Adaptive scaling factors
    F = 0.3 + 0.5 * exp(-2 * w_f .* w_c);
    F = repmat(F, 1, D);
    
    % Generate 6 random indices per individual
    idx = arrayfun(@(x) randperm(NP, 6), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3); 
    r4 = idx(:,4); r5 = idx(:,5); r6 = idx(:,6);
    
    % Enhanced directional mutation
    base = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    diff3 = popdecs(r5,:) - popdecs(r6,:);
    mutant = base + F .* diff1 + 0.7 * diff2 + 0.3 * diff3;
    
    % Adaptive crossover rate
    CR = 0.9 - 0.5 * sqrt(w_f + w_c);
    CR = max(min(CR, 0.95), 0.05);
    
    % Binomial crossover with guaranteed change
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    delta = 0.02 * (ub_rep - lb_rep) .* rand(NP,D);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = (lb_rep(below_lb) + popdecs(below_lb))/2 + delta(below_lb);
    offspring(above_ub) = (ub_rep(above_ub) + popdecs(above_ub))/2 - delta(above_ub);
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
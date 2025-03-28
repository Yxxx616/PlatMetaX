% MATLAB Code
function [offspring] = updateFunc726(popdecs, popfits, cons)
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
    F = 0.5 + 0.3 * (w_f .* w_c);
    F = repmat(F, 1, D);
    
    % Generate random indices (4 per individual)
    idx = arrayfun(@(x) randperm(NP, 4), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3); r4 = idx(:,4);
    
    % Directional mutation
    base = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutant = base + F .* diff1 + 0.5 * diff2;
    
    % Adaptive crossover rate
    CR = 0.9 - 0.4 * sqrt(w_f .* w_c);
    CR = max(min(CR, 0.95), 0.05);
    
    % Binomial crossover with guaranteed change
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    delta = 0.01 * (ub_rep - lb_rep) .* rand(NP,D);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = (lb_rep(below_lb) + popdecs(below_lb))/2 + delta(below_lb);
    offspring(above_ub) = (ub_rep(above_ub) + popdecs(above_ub))/2 - delta(above_ub);
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
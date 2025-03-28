% MATLAB Code
function [offspring] = updateFunc1320(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Constraint-aware best selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_best = popdecs(temp(best_idx), :);
    else
        [~, best_idx] = min(abs(cons));
        x_best = popdecs(best_idx, :);
    end
    
    % 2. Adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    
    norm_f = (popfits - f_min) ./ (f_max - f_min + eps_val);
    norm_c = abs(cons) ./ (c_max + eps_val);
    F = 0.4 + 0.5 * (1 - norm_f) .* (1 - norm_c);
    F = F(:, ones(1, D));
    
    % 3. Directional mutation
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(NP+1:2*NP);
    r2(r2 == r1) = mod(r2(r2 == r1) + 1;
    
    mutants = popdecs + F .* (x_best - popdecs) + ...
              F .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 4. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.85 - 0.35*(rank/NP);
    CR = CR(:, ones(1,D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Boundary handling with reflection
    reflect_prob = 0.6;
    do_reflect = rand(NP, D) < reflect_prob;
    
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    
    % Apply reflection or clamping
    offspring(lb_mask & do_reflect) = 2*lb(lb_mask & do_reflect) - offspring(lb_mask & do_reflect);
    offspring(ub_mask & do_reflect) = 2*ub(ub_mask & do_reflect) - offspring(ub_mask & do_reflect);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
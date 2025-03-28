% MATLAB Code
function [offspring] = updateFunc1317(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify best solution considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_best = popdecs(temp(best_idx), :);
    else
        [~, best_idx] = min(abs(cons));
        x_best = popdecs(best_idx, :);
    end
    
    % 2. Calculate constraint weights and direction vectors
    abs_cons = abs(cons);
    weights = 1./(1 + abs_cons);
    v = popdecs + weights.*(x_best - popdecs);
    
    % 3. Compute adaptive coefficients
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs_cons);
    alpha = 0.5 * (1 + (popfits - f_min)./(f_max - f_min + eps));
    beta = 0.5 * (1 - abs_cons./c_max);
    
    % 4. Generate random indices for differential vectors
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(NP+1:2*NP);
    r2(r2 == r1) = mod(r2(r2 == r1) + randi(NP-1), NP) + 1;
    
    % 5. Create mutation vectors
    F = alpha.*beta;
    rand_vec = rand(NP, D);
    mutants = v + F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
              (1-alpha).*rand_vec.*(x_best - popdecs);
    
    % 6. Rank-based crossover probability
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.4*(rank/NP);
    CR = CR(:, ones(1,D));
    
    % 7. Perform crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with adaptive reflection
    reflect_prob = 0.7 - 0.4*(rank/NP);
    reflect_prob = reflect_prob(:, ones(1,D));
    do_reflect = rand(NP, D) < reflect_prob;
    
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    
    offspring(lb_mask & do_reflect) = 2*lb(lb_mask & do_reflect) - offspring(lb_mask & do_reflect);
    offspring(ub_mask & do_reflect) = 2*ub(ub_mask & do_reflect) - offspring(ub_mask & do_reflect);
    offspring(~do_reflect) = min(max(offspring(~do_reflect), lb(~do_reflect)), ub(~do_reflect));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
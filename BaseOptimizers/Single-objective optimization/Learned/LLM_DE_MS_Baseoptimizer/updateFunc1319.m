% MATLAB Code
function [offspring] = updateFunc1319(popdecs, popfits, cons)
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
    
    % 2. Calculate adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    eps_val = 1e-10;
    
    F_f = 0.5 + 0.5 * (popfits - f_min) ./ (f_max - f_min + eps_val);
    F_c = 0.5 * (1 - abs(cons) ./ (c_max + eps_val));
    F = F_f .* F_c;
    
    % 3. Generate random indices for differential vectors
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(NP+1:2*NP);
    r2(r2 == r1) = mod(r2(r2 == r1) + randi(NP-1), NP) + 1;
    
    % 4. Create mutation vectors with direction guidance
    mutants = popdecs + F .* (x_best - popdecs) + ...
              F .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 5. Rank-based crossover probability
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.4*(rank/NP);
    CR = CR(:, ones(1,D));
    
    % 6. Perform crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with adaptive reflection
    reflect_prob = 0.7 - 0.4*(rank/NP);
    reflect_prob = reflect_prob(:, ones(1,D));
    do_reflect = rand(NP, D) < reflect_prob;
    
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    
    % Apply reflection or clamping
    offspring(lb_mask & do_reflect) = 2*lb(lb_mask & do_reflect) - offspring(lb_mask & do_reflect);
    offspring(ub_mask & do_reflect) = 2*ub(ub_mask & do_reflect) - offspring(ub_mask & do_reflect);
    offspring(~do_reflect) = min(max(offspring(~do_reflect), lb(~do_reflect)), ub(~do_reflect));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
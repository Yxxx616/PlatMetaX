% MATLAB Code
function [offspring] = updateFunc1314(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with improved feasibility handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, best_fit_idx] = min(popfits);
        [~, best_cons_idx] = min(abs(cons));
        x_elite = 0.7*popdecs(best_fit_idx,:) + 0.3*popdecs(best_cons_idx,:);
    end
    
    % 2. Constraint-weighted best individuals
    abs_cons = abs(cons);
    [sorted_cons, cons_idx] = sort(abs_cons);
    top_k = min(5, NP);
    x_best_cons = popdecs(cons_idx(1:top_k), :);
    cons_weights = 1./(1 + sorted_cons(1:top_k)');
    cons_weights = cons_weights/sum(cons_weights);
    x_best_cons_weighted = sum(bsxfun(@times, x_best_cons, cons_weights), 1);
    
    % 3. Rank-based differential vectors
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    prob = 1 - rank/NP;
    
    % Select r1 and r2 based on rank probability
    r1 = arrayfun(@(i) find(rand() < cumsum(prob), 1:NP)');
    r2 = arrayfun(@(i) find(rand() < cumsum(prob), 1:NP)');
    while any(r1 == r2)
        r2 = arrayfun(@(i) find(rand() < cumsum(prob), 1:NP)');
    end
    d_rank = popdecs(r1,:) - popdecs(r2,:);
    
    % 4. Compute directions
    d_elite = bsxfun(@minus, x_elite, popdecs);
    d_cons = bsxfun(@minus, x_best_cons_weighted, popdecs);
    
    % 5. Adaptive scaling factors
    F_base = 0.4 + 0.4*(rank/NP);
    norm_cons = (abs_cons - min(abs_cons)) ./ (max(abs_cons) - min(abs_cons) + eps);
    F_cons = 0.5*(1 - exp(-5*norm_cons));
    
    % 6. Combined mutation
    F_base = F_base(:, ones(1,D));
    F_cons = F_cons(:, ones(1,D));
    mutants = popdecs + F_base.*(d_elite + d_cons) + F_cons.*d_rank;
    
    % 7. Rank-based crossover
    CR = 0.85 - 0.35*(rank/NP);
    CR = CR(:, ones(1,D));
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with adaptive reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    reflect_prob = 0.7 + 0.2*(rank/NP);
    reflect_prob = reflect_prob(:, ones(1,D));
    do_reflect = rand(NP, D) < reflect_prob;
    
    offspring(lb_mask & do_reflect) = 2*lb(lb_mask & do_reflect) - offspring(lb_mask & do_reflect);
    offspring(ub_mask & do_reflect) = 2*ub(ub_mask & do_reflect) - offspring(ub_mask & do_reflect);
    offspring(~do_reflect) = min(max(offspring(~do_reflect), lb(~do_reflect)), ub(~do_reflect));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
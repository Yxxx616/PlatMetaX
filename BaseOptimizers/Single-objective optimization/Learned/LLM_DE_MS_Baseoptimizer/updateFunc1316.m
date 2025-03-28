% MATLAB Code
function [offspring] = updateFunc1316(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-weighted centroid
    abs_cons = abs(cons);
    weights = 1./(1 + abs_cons);
    weighted_sum = sum(bsxfun(@times, popdecs, weights), 1);
    centroid = weighted_sum / sum(weights);
    
    % 2. Elite selection with improved feasibility handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, best_cons_idx] = min(abs_cons);
        x_elite = popdecs(best_cons_idx, :);
    end
    
    % 3. Opposition-based population
    x_opp = bsxfun(@plus, lb, ub) - popdecs;
    
    % 4. Adaptive scaling factors
    max_cons = max(abs_cons);
    F = 0.4 + 0.4 * (1 - abs_cons/max_cons);
    F = F(:, ones(1,D));
    
    % 5. Rank-based probability
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    
    % 6. Generate mutants
    d_elite = bsxfun(@minus, x_elite, popdecs);
    d_centroid = bsxfun(@minus, centroid, x_opp);
    mutants = popdecs + F.*d_elite + 0.5*d_centroid;
    
    % 7. Constraint-aware crossover
    CR = 0.9 - 0.5*(rank/NP);
    CR = CR(:, ones(1,D));
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
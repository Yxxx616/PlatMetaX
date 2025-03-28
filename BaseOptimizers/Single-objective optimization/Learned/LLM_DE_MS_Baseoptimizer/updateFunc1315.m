% MATLAB Code
function [offspring] = updateFunc1315(popdecs, popfits, cons)
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
        x_elite = 0.6*popdecs(best_fit_idx,:) + 0.4*popdecs(best_cons_idx,:);
    end
    
    % 2. Constraint-weighted best individuals
    abs_cons = abs(cons);
    [sorted_cons, cons_idx] = sort(abs_cons);
    top_k = min(5, NP);
    x_best_cons = popdecs(cons_idx(1:top_k), :);
    cons_weights = 1./(1 + sorted_cons(1:top_k)');
    cons_weights = cons_weights/sum(cons_weights);
    x_best_cons_weighted = sum(bsxfun(@times, x_best_cons, cons_weights), 1);
    
    % 3. Rank-based probability
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    prob = 1 - rank/NP;
    prob = prob/sum(prob);
    
    % 4. Constraint-aware scaling factors
    norm_cons = (abs_cons - min(abs_cons)) ./ (max(abs_cons) - min(abs_cons) + eps);
    F = 0.5 + 0.3 * tanh(5 * (1 - norm_cons));
    F = F(:, ones(1,D));
    
    % 5. Generate direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    d_cons = bsxfun(@minus, x_best_cons_weighted, popdecs);
    
    % 6. Rank-based differential vectors
    r1 = randsample(NP, NP, true, prob);
    r2 = randsample(NP, NP, true, prob);
    while any(r1 == r2)
        r2 = randsample(NP, NP, true, prob);
    end
    d_rank = popdecs(r1,:) - popdecs(r2,:);
    
    % 7. Combined mutation
    mutants = popdecs + F.*(d_elite + 0.7*d_cons) + 0.3*F.*d_rank;
    
    % 8. Adaptive crossover
    CR = 0.9 - 0.5*(rank/NP);
    CR = CR(:, ones(1,D));
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    reflect_prob = 0.8 - 0.3*(rank/NP);
    reflect_prob = reflect_prob(:, ones(1,D));
    do_reflect = rand(NP, D) < reflect_prob;
    
    offspring(lb_mask & do_reflect) = 2*lb(lb_mask & do_reflect) - offspring(lb_mask & do_reflect);
    offspring(ub_mask & do_reflect) = 2*ub(ub_mask & do_reflect) - offspring(ub_mask & do_reflect);
    offspring(~do_reflect) = min(max(offspring(~do_reflect), lb(~do_reflect)), ub(~do_reflect));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
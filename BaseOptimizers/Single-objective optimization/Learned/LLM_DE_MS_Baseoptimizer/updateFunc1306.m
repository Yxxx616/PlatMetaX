% MATLAB Code
function [offspring] = updateFunc1306(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, best_fit_idx] = min(popfits);
        [~, best_cons_idx] = min(cons);
        x_elite = 0.7*popdecs(best_fit_idx,:) + 0.3*popdecs(best_cons_idx,:);
    end
    
    % 2. Opposition-based vectors
    x_opp = lb + ub - popdecs;
    
    % 3. Constraint-aware direction
    [~, best_cons_idx] = min(cons);
    x_best_cons = popdecs(best_cons_idx, :);
    d_cons = sign(cons) .* bsxfun(@minus, x_best_cons, popdecs);
    
    % 4. Elite direction
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    % 5. Adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    min_cons = min(cons);
    max_cons = max(cons);
    
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    w_c = (cons - min_cons) / (max_cons - min_cons + eps);
    w = 1 ./ (1 + exp(-5*(w_f - w_c)));
    w = w(:, ones(1,D));
    
    % 6. Adaptive scaling factor
    F = 0.5 + 0.3 * rand(NP,1) .* (1 - w(:,1));
    F = min(max(F, 0.3), 0.9);
    F = F(:, ones(1,D));
    
    % 7. Combined mutation
    mutants = popdecs + F .* (w.*d_elite + (1-w).*(x_opp - popdecs + d_cons));
    
    % 8. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.5*(rank/NP);
    CR = CR(:, ones(1,D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
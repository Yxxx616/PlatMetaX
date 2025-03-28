% MATLAB Code
function [offspring] = updateFunc1313(popdecs, popfits, cons)
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
        alpha = 0.6 + 0.3*rand();
        x_elite = alpha*popdecs(best_fit_idx,:) + (1-alpha)*popdecs(best_cons_idx,:);
    end
    
    % 2. Constraint-aware direction with adaptive scaling
    [~, best_cons_idx] = min(abs(cons));
    x_best_cons = popdecs(best_cons_idx, :);
    cons_sign = sign(cons);
    d_cons = cons_sign(:, ones(1,D)) .* bsxfun(@minus, x_best_cons, popdecs);
    
    % 3. Elite-guided direction
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    % 4. Random differential vectors with unique indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Adaptive weights based on normalized fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    abs_cons = abs(cons);
    min_cons = min(abs_cons);
    max_cons = max(abs_cons);
    
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    w_c = (abs_cons - min_cons) / (max_cons - min_cons + eps);
    w = 1 ./ (1 + exp(-5*(w_f - w_c)));
    w = w(:, ones(1,D));
    
    % 6. Rank-based scaling factor (more aggressive than previous version)
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    F = 0.2 + 0.5*(rank/NP);
    F = F(:, ones(1,D));
    
    % 7. Combined mutation with reduced noise
    mutants = popdecs + F .* (w.*d_elite + (1-w).*d_cons) + 0.2*randn(NP,D).*d_rand;
    
    % 8. Rank-based crossover probability (more exploration for worse solutions)
    CR = 0.9 - 0.4*(rank/NP);
    CR = CR(:, ones(1,D));
    
    % Crossover operation
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with probabilistic reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    reflect_prob = rand(NP, D) < 0.8;
    offspring(lb_mask & reflect_prob) = 2*lb(lb_mask & reflect_prob) - offspring(lb_mask & reflect_prob);
    offspring(ub_mask & reflect_prob) = 2*ub(ub_mask & reflect_prob) - offspring(ub_mask & reflect_prob);
    offspring(~reflect_prob) = min(max(offspring(~reflect_prob), lb(~reflect_prob)), ub(~reflect_prob));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
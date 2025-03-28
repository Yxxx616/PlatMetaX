% MATLAB Code
function [offspring] = updateFunc1202(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    infeas_mask = ~feasible_mask;
    if any(infeas_mask)
        x_inf = mean(popdecs(infeas_mask, :), 1);
    else
        x_inf = zeros(1, D);
    end
    
    % Find most diverse individual
    centroid = mean(popdecs, 1);
    dist = sqrt(sum((popdecs - centroid).^2, 2));
    [~, div_idx] = max(dist);
    x_div = popdecs(div_idx, :);
    
    % 2. Calculate adaptive weights
    f_min = min(popfits);
    f_max = max(popfits);
    alpha = (popfits - f_min) ./ (f_max - f_min + eps);
    
    c_max = max(abs(cons));
    beta = exp(-abs(cons) ./ (c_max + eps));
    
    [~, dist_rank] = sort(dist, 'descend');
    gamma = (dist_rank ./ NP)';
    
    % 3. Generate mutation vectors
    F = 0.5 + 0.2 * randn(NP, 1);
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    F1 = alpha .* beta;
    F2 = (1-alpha) .* beta;
    F3 = (1-beta) .* gamma;
    
    term1 = F1 .* (x_best - popdecs);
    term2 = F2 .* (x_feas - x_inf);
    term3 = F3 .* (x_div - popdecs);
    term4 = F .* (popdecs(r1,:) - popdecs(r2,:));
    
    mutants = popdecs + term1 + term2 + term3 + term4;
    
    % 4. Adaptive crossover with jitter
    CR = 0.2 + 0.5 * alpha;
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
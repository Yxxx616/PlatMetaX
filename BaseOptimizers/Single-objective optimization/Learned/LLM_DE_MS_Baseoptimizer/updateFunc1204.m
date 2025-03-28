% MATLAB Code
function [offspring] = updateFunc1204(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Feasible and infeasible centroids
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    infeas = ~feasible;
    if any(infeas)
        x_inf = mean(popdecs(infeas, :), 1);
    else
        x_inf = zeros(1, D);
    end
    
    % 2. Compute diversity ranks
    centroid = mean(popdecs, 1);
    dist = sqrt(sum((popdecs - centroid).^2, 2));
    [~, dist_rank] = sort(dist, 'descend');
    gamma = dist_rank / NP;
    
    % 3. Adaptive weights
    f_min = min(popfits);
    f_max = max(popfits);
    alpha = (popfits - f_min) ./ (f_max - f_min + eps);
    
    c_max = max(abs(cons));
    beta = exp(-abs(cons) ./ (c_max + eps));
    
    % 4. Mutation components
    F1 = 0.8 .* alpha .* beta;
    F2 = 0.6 .* (1-alpha) .* beta;
    F3 = 0.4 .* (1-beta) .* gamma;
    
    % Random indices for differential component
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    % Mutation vectors
    term1 = F1 .* (x_best - popdecs);
    term2 = F2 .* (x_feas - x_inf);
    term3 = F3 .* (popdecs(r1,:) - popdecs(r2,:));
    
    mutants = popdecs + term1 + term2 + term3;
    
    % 5. Adaptive crossover
    CR = 0.1 + 0.7 .* alpha .* beta;
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
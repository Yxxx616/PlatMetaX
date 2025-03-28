% MATLAB Code
function [offspring] = updateFunc1564(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-10;
    
    % 1. Normalize constraints and fitness
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons);
    norm_cons = (abs_cons - mean_cons) ./ (std_cons + eps);
    
    mean_fit = mean(popfits);
    std_fit = std(popfits);
    norm_fits = (popfits - mean_fit) ./ (std_fit + eps);
    
    % 2. Identify key solutions
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 3. Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    r3 = randi(NP, NP, 1);
    
    % 4. Calculate weights
    w_cons = 1 ./ (1 + exp(-5 * norm_cons));
    w_fit = 1 ./ (1 + exp(-5 * norm_fits));
    
    % 5. Base vector selection
    x_rand = popdecs(r3, :);
    x_base = bsxfun(@times, w_cons, x_best) + ...
             bsxfun(@times, (1-w_cons).*w_fit, x_feas) + ...
             bsxfun(@times, (1-w_cons).*(1-w_fit), x_rand);
    
    % 6. Direction vectors
    d_best = bsxfun(@minus, x_best, popdecs);
    d_feas = bsxfun(@minus, x_feas, popdecs);
    d_rand = bsxfun(@minus, x_rand, popdecs);
    
    % 7. Adaptive scaling factors
    F = 0.5 + 0.3 * tanh(norm_fits);
    G = 0.2 + 0.1 * tanh(norm_cons);
    
    % 8. Mutation
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    mutants = x_base + bsxfun(@times, F, (0.6*d_best + 0.3*d_feas + 0.1*d_rand)) + ...
              bsxfun(@times, G, rand_diff);
    
    % 9. Dynamic crossover
    CR = 0.7 + 0.2 * tanh(norm_cons);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % 10. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 11. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 12. Elite refinement
    [~, sorted_idx] = sort(popfits);
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb) .* (1 - w_cons(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1565(popdecs, popfits, cons)
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
    
    % 3. Generate random indices ensuring r1 ≠ r2 ≠ i
    r1 = arrayfun(@(i) setdiff(randperm(NP, 2), i), 1:NP, 'UniformOutput', false);
    r1 = cell2mat(r1)';
    r2 = r1(:,2);
    r1 = r1(:,1);
    r3 = randi(NP, NP, 1);
    
    % 4. Calculate adaptive weights
    w_fit = 1 ./ (1 + exp(-5 * norm_fits));
    w_cons = 1 ./ (1 + exp(5 * norm_cons));
    w_rand = 1 - w_fit - w_cons;
    
    % 5. Base vector construction
    x_rand = popdecs(r3, :);
    x_base = bsxfun(@times, w_fit, x_best) + ...
             bsxfun(@times, w_cons, x_feas) + ...
             bsxfun(@times, w_rand, x_rand);
    
    % 6. Direction vectors with adaptive coefficients
    d_best = bsxfun(@minus, x_best, popdecs);
    d_feas = bsxfun(@minus, x_feas, popdecs);
    d_rand = bsxfun(@minus, x_rand, popdecs);
    
    alpha = 0.6 * (1 - w_cons);
    beta = 0.3 * w_cons;
    gamma = 0.1 * (1 - w_fit);
    
    d_adapt = bsxfun(@times, alpha, d_best) + ...
              bsxfun(@times, beta, d_feas) + ...
              bsxfun(@times, gamma, d_rand);
    
    % 7. Adaptive scaling factors
    F = 0.5 + 0.3 * tanh(norm_fits);
    G = 0.2 + 0.1 * tanh(norm_cons);
    
    % 8. Hybrid mutation
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    mutants = x_base + bsxfun(@times, F, d_adapt) + ...
              bsxfun(@times, G, rand_diff);
    
    % 9. Dynamic crossover with constraint awareness
    CR = 0.7 + 0.2 * tanh(norm_cons);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % 10. Create offspring with boundary reflection
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
    
    % 11. Elite refinement for top 10%
    [~, sorted_idx] = sort(popfits);
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.01 * (ub - lb) .* (1 - w_cons(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
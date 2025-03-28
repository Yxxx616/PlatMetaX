% MATLAB Code
function [offspring] = updateFunc1567(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-10;
    
    % 1. Normalize fitness and constraints
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - mean_fit) ./ std_fit;
    
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons) + eps;
    norm_cons = (abs_cons - mean_cons) ./ std_cons;
    
    % 2. Identify key solutions
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 3. Generate random indices (r1 ≠ r2 ≠ i)
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        selected = available(randperm(length(available), 2));
        r1(i) = selected(1);
        r2(i) = selected(2);
    end
    
    % 4. Calculate adaptive weights
    w_fit = 1 ./ (1 + exp(-5 * norm_fits));
    w_cons = 1 ./ (1 + exp(5 * norm_cons));
    w_rand = max(0, 1 - w_fit - w_cons);
    
    % 5. Base vectors
    x_rand = popdecs(randi(NP, NP, 1), :);
    x_base = bsxfun(@times, w_fit, x_best) + ...
             bsxfun(@times, w_cons, x_feas) + ...
             bsxfun(@times, w_rand, x_rand);
    
    % 6. Direction vectors
    d_best = bsxfun(@minus, x_best, popdecs);
    d_feas = bsxfun(@minus, x_feas, popdecs);
    
    % 7. Adaptive scaling factors
    F = 0.5 + 0.3 * tanh(norm_cons);
    
    % 8. Differential vectors
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    
    % 9. Mutation
    mutants = x_base + bsxfun(@times, F, diff_vec) + ...
              0.5 * bsxfun(@times, w_fit, d_best) + ...
              0.3 * bsxfun(@times, w_cons, d_feas);
    
    % 10. Crossover with adaptive CR
    CR = 0.7 + 0.2 * tanh(norm_cons);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
    
    % Elite refinement for top solutions
    [~, sorted_idx] = sort(popfits);
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.01 * (ub - lb) .* (1 - w_cons(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
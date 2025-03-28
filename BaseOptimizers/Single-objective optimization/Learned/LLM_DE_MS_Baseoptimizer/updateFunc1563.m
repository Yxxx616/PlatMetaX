% MATLAB Code
function [offspring] = updateFunc1563(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-10;
    
    % 1. Normalization
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons);
    norm_cons = (abs_cons - mean_cons) ./ (std_cons + eps);
    
    mean_fit = mean(popfits);
    std_fit = std(popfits);
    norm_fits = (popfits - mean_fit) ./ (std_fit + eps);
    
    % 2. Weight calculation
    w = 1 ./ (1 + exp(-5 * norm_cons));  % Constraint weight
    alpha = 1 ./ (1 + exp(-5 * norm_fits));  % Fitness weight
    
    % 3. Best and feasible solutions
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 4. Base vectors
    x_base = bsxfun(@times, w, x_best) + bsxfun(@times, (1-w), x_feas);
    
    % 5. Direction vectors
    best_dir = bsxfun(@minus, x_best, popdecs);
    feas_dir = bsxfun(@minus, x_feas, popdecs);
    dir_combined = bsxfun(@times, alpha, best_dir) + bsxfun(@times, (1-alpha), feas_dir);
    
    % 6. Adaptive scaling factors
    F = 0.5 + 0.3 * tanh(norm_fits);
    
    % 7. Novel differential with two random pairs
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    while any(r3 == r4)
        r4 = randi(NP, NP, 1);
    end
    rand_diff1 = popdecs(r1,:) - popdecs(r2,:);
    rand_diff2 = popdecs(r3,:) - popdecs(r4,:);
    
    % 8. Mutation
    mutants = x_base + bsxfun(@times, F, dir_combined) + ...
              0.1 * rand_diff1 + 0.05 * rand_diff2;
    
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
    top_N = max(1, round(0.2*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.02 * (ub - lb) .* (1 - w(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
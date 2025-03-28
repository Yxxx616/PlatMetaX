% MATLAB Code
function [offspring] = updateFunc1556(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-10;
    
    % 1. Constraint-aware parameters
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons);
    norm_cons = (abs_cons - mean_cons) ./ (std_cons + eps);
    F = 0.2 + 0.5 * (1 - tanh(5 * norm_cons));
    w = 1 ./ (1 + exp(-5 * norm_cons));
    
    % 2. Elite selection (top 20%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.2 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    elite_idx = randi(elite_N, NP, 1);
    elite_dir = elite_pool(elite_idx,:) - popdecs;
    
    % 3. Feasible attraction direction
    feasible = cons <= 0;
    if any(feasible)
        mean_feas = mean(popdecs(feasible,:), 1);
    else
        mean_feas = mean(popdecs, 1);
    end
    feas_dir = bsxfun(@minus, mean_feas, popdecs);
    
    % 4. Hybrid mutation
    dir_combined = bsxfun(@times, w, feas_dir) + bsxfun(@times, (1-w), elite_dir);
    mutants = popdecs + bsxfun(@times, F, dir_combined);
    
    % 5. Adaptive crossover
    max_cons = max(abs_cons);
    CR = 0.1 + 0.7 * (1 - abs_cons./(max_cons + eps));
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = (lb(mask_lb) + popdecs(mask_lb)) / 2;
    offspring(mask_ub) = (ub(mask_ub) + popdecs(mask_ub)) / 2;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.005 * (ub - lb) .* (1 - w(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
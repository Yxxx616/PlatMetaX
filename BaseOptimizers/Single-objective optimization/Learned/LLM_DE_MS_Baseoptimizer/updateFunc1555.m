% MATLAB Code
function [offspring] = updateFunc1555(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-6;
    
    % 1. Constraint-aware weights
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons);
    w = 1 ./ (1 + exp(-5*(abs_cons - mean_cons)./max(std_cons, eps)));
    
    % 2. Elite selection (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.3 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    
    % 3. Feasible centroid direction
    feasible = cons <= 0;
    if any(feasible)
        mean_feas = mean(popdecs(feasible,:), 1);
    else
        mean_feas = mean(popdecs, 1);
    end
    feas_dir = bsxfun(@minus, mean_feas, popdecs);
    
    % 4. Random differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2 | r1 == (1:NP)' | r2 == (1:NP)')
        r1 = randi(NP, NP, 1);
        r2 = randi(NP, NP, 1);
    end
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Elite indices
    elite_idx = randi(elite_N, NP, 1);
    elite_dir = elite_pool(elite_idx,:) - popdecs;
    
    % 5. Adaptive mutation
    F = 0.3 + 0.5 * (1 - w);
    dir_combined = bsxfun(@times, w, feas_dir) + ...
                  bsxfun(@times, (1-w), 0.6*elite_dir + 0.4*rand_dir);
    mutants = popdecs + bsxfun(@times, F, dir_combined);
    
    % 6. Dynamic crossover
    CR = 0.2 + 0.6 * w;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
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
    top_N = max(1, round(0.15*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb) .* (1 - w(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
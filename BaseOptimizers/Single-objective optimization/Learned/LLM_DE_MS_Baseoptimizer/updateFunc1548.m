% MATLAB Code
function [offspring] = updateFunc1548(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Calculate feasibility weights
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    alpha = 1 ./ (1 + exp(-5 * (abs_cons - mean_cons)));
    
    % 2. Select elite solutions (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.3 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    
    % 3. Calculate feasible centroid
    feasible = cons <= 0;
    if any(feasible)
        mean_feas = mean(popdecs(feasible,:), 1);
    else
        mean_feas = mean(popdecs, 1);
    end
    
    % 4. Opposition-based direction
    opposite_dir = (lb + ub - popdecs) - popdecs;
    
    % 5. Calculate directional vectors
    elite_dir = elite - popdecs;
    feas_dir = mean_feas - popdecs;
    
    % 6. Adaptive weights
    w1 = 0.6 * (1 - alpha);
    w2 = 0.3 * alpha;
    w3 = 0.1 * ones(NP, 1);
    
    % 7. Generate mutation vectors
    mutants = popdecs + w1.*elite_dir + w2.*feas_dir + w3.*opposite_dir;
    
    % 8. Dynamic crossover
    CR = 0.2 + 0.6 * alpha;
    mask_cr = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 9. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = lb(mask_lb) + 0.5 * rand(sum(sum(mask_lb)),1) .* (popdecs(mask_lb) - lb(mask_lb));
    offspring(mask_ub) = ub(mask_ub) - 0.5 * rand(sum(sum(mask_ub)),1) .* (ub(mask_ub) - popdecs(mask_ub));
    
    % 10. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 11. Local refinement for top 15% solutions
    top_N = max(1, round(0.15*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
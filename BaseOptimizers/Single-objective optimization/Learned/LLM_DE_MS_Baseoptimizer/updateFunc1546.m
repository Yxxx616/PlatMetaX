% MATLAB Code
function [offspring] = updateFunc1546(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Calculate feasibility ratio
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
    
    % 4. Opposition-based population
    opposite_pop = lb + ub - popdecs;
    
    % 5. Calculate weights
    w1 = 0.5 * (1 - alpha);
    w2 = 0.3 * alpha;
    w3 = 0.2 * (1 - alpha);
    
    % 6. Generate mutation vectors
    elite_dir = elite - popdecs;
    feas_dir = mean_feas - popdecs;
    opp_dir = opposite_pop - popdecs;
    
    mutants = popdecs + w1.*elite_dir + w2.*feas_dir + w3.*opp_dir;
    
    % 7. Adaptive crossover with dynamic CR
    CR = 0.1 + 0.7 * alpha;
    mask_cr = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 8. Boundary handling with adaptive reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = lb(mask_lb) + 0.5 * rand(sum(sum(mask_lb)),1) .* (popdecs(mask_lb) - lb(mask_lb));
    offspring(mask_ub) = ub(mask_ub) - 0.5 * rand(sum(sum(mask_ub)),1) .* (ub(mask_ub) - popdecs(mask_ub));
    
    % 9. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 10. Local refinement for top solutions
    top_N = max(1, round(0.15*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
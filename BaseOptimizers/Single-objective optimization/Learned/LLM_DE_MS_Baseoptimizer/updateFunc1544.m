% MATLAB Code
function [offspring] = updateFunc1544(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Normalize fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fit = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_cons = max(abs_cons);
    norm_cons = abs_cons / (max_cons + eps);
    
    % 2. Calculate adaptive weights
    alpha = 0.3 * norm_fit + 0.7 * norm_cons;
    w1 = 0.5 * (1 - alpha);
    w2 = 0.3 * alpha;
    w3 = 0.2 * ones(NP, 1);
    
    % 3. Select elite solutions (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.3 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    
    % 4. Calculate constraint-aware direction
    feasible = cons <= 0;
    if any(feasible)
        mean_feas = mean(popdecs(feasible,:), 1);
    else
        mean_feas = elite_pool(1,:);
    end
    constraint_dir = mean_feas - popdecs;
    
    % 5. Generate differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % 6. Create mutation vectors
    elite_dir = elite - popdecs;
    sigma = 0.1 * alpha;
    mutants = popdecs + w1.*elite_dir + w2.*constraint_dir + w3.*diff_dir;
    mutants = mutants + sigma.*randn(NP,1).*randn(NP,D);
    
    % 7. Adaptive crossover
    CR = 0.1 + 0.8 * alpha;
    mask_cr = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 8. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % 9. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 10. Local refinement for top solutions
    top_N = max(1, round(0.15*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.05 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
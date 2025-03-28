% MATLAB Code
function [offspring] = updateFunc1518(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Calculate constraint violations
    cv = max(0, cons);
    feasible_mask = cv == 0;
    num_feas = sum(feasible_mask);
    
    % Normalize constraint violations
    norm_cv = cv / (max(cv) + eps);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * (1 - exp(-5 * norm_cv));
    CR = 0.9 - 0.5 * norm_cv;
    
    % Select pbest individuals (top 30%)
    [~, sorted_idx] = sort(popfits);
    pbest_size = max(1, round(0.3 * NP));
    pbest_pool = sorted_idx(1:pbest_size);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    % Ensure r1 != r2
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    
    % Select pbest for each individual
    pbest_idx = pbest_pool(randi(pbest_size, NP, 1));
    
    % Select feasible individuals if available
    if num_feas > 0
        feas_pool = find(feasible_mask);
        feas_idx = feas_pool(randi(num_feas, NP, 1));
    else
        feas_idx = randi(NP, NP, 1);
    end
    
    % Hybrid mutation
    term1 = popdecs(pbest_idx,:) - popdecs;
    term2 = popdecs(feas_idx,:) - popdecs;
    term3 = popdecs(r1,:) - popdecs(r2,:);
    
    mutants = popdecs + F(:, ones(1,D)) .* (term1 + term2 + term3)/3;
    
    % Crossover with adaptive CR
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % Boundary handling with mixed strategy
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    
    % Reflection for most cases
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Random reset for 10% of violations
    reset_mask = rand(NP,D) < 0.1;
    reset_lb = mask_lb & reset_mask;
    reset_ub = mask_ub & reset_mask;
    
    offspring(reset_lb) = lb(reset_lb) + rand(sum(reset_lb(:)),1).*(ub(reset_lb)-lb(reset_lb));
    offspring(reset_ub) = lb(reset_ub) + rand(sum(reset_ub(:)),1).*(ub(reset_ub)-lb(reset_ub));
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % Local search for top solutions
    [~, elite_idx] = sort(popfits + 1e6*cv);
    elite_N = max(1, round(0.2*NP));
    elite = elite_idx(1:elite_N);
    
    sigma = 0.1 * (1 - norm_cv(elite, ones(1,D))) .* (ub - lb);
    local_search = popdecs(elite,:) + sigma .* randn(elite_N, D);
    local_search = min(max(local_search, lb), ub);
    
    offspring(elite,:) = local_search;
end
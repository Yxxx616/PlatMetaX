% MATLAB Code
function [offspring] = updateFunc1516(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Elite selection with constraint handling
    penalty = 1e6 * max(0, cons);
    [~, elite_idx] = min(popfits + penalty);
    x_elite = popdecs(elite_idx, :);
    
    % Normalize constraints
    cv = max(0, cons);
    norm_cv = cv / (max(cv) + eps);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(norm_cv);
    CR = 0.9 - 0.5 * norm_cv;
    sigma = 0.2 * (1 - norm_cv) .* (ub - lb);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP);
    
    % Mutation with directional information
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    mutants = x_elite(ones(NP,1),:) + F(:, ones(1,D)).*diff_vec + ...
              sigma(:, ones(1,D)).*randn(NP, D);
    
    % Crossover
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % Boundary handling with 10% random reset
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    mask_reset = rand(NP,D) < 0.1;
    
    % Apply boundary handling
    offspring = min(max(offspring, lb), ub);
    % Random reset for 10% of out-of-bounds variables
    offspring(mask_lb & mask_reset) = lb(mask_lb & mask_reset) + ...
        rand(sum(sum(mask_lb & mask_reset)),1).*(ub(mask_lb & mask_reset)-lb(mask_lb & mask_reset));
    offspring(mask_ub & mask_reset) = lb(mask_ub & mask_reset) + ...
        rand(sum(sum(mask_ub & mask_reset)),1).*(ub(mask_ub & mask_reset)-lb(mask_ub & mask_reset));
    
    % Local search for top solutions
    [~, sorted_idx] = sort(popfits + penalty);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.1 * (1 - norm_cv(refine_idx, ones(1,D))) .* (ub - lb);
    local_search = popdecs(refine_idx,:) + sigma_refine .* randn(refine_N, D);
    local_search = min(max(local_search, lb), ub);
    
    % Combine with main offspring
    offspring(refine_idx,:) = local_search;
end
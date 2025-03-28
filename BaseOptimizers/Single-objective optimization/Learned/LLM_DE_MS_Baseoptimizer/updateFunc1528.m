% MATLAB Code
function [offspring] = updateFunc1528(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Calculate normalized constraint violations
    cv = max(0, cons);
    norm_cv = cv / (max(cv) + eps);
    
    % Calculate fitness ranks (0 for best, 1 for worst)
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = (0:NP-1)'/(NP-1);
    
    % Adaptive scaling factor
    F = 0.5 + 0.3 * (1 - norm_cv);
    
    % Adaptive crossover rate
    CR = 0.9 - 0.5 * norm_cv;
    
    % Identify feasible solutions
    feasible_mask = cv == 0;
    num_feas = sum(feasible_mask);
    
    % Select elite pool (top 20%)
    elite_size = max(1, round(0.2 * NP));
    elite_pool = sorted_idx(1:elite_size);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    % Ensure r1 != r2
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    
    % Select feasible individuals if available
    if num_feas > 0
        feas_pool = find(feasible_mask);
        feas_idx = feas_pool(randi(num_feas, NP, 1));
    else
        feas_idx = elite_pool(randi(elite_size, NP, 1));
    end
    
    % Select elite individuals
    elite_idx = elite_pool(randi(elite_size, NP, 1));
    
    % Calculate adaptive weights
    denom = 3 + (1 - norm_cv) + (1 - ranks);
    w_feas = 2 * (1 - norm_cv) ./ denom;
    w_elite = (1 + (1 - ranks)) ./ denom;
    w_div = 1 ./ denom;
    
    % Compute mutation vectors
    v_feas = popdecs(feas_idx,:) - popdecs;
    v_elite = popdecs(elite_idx,:) - popdecs;
    v_div = popdecs(r1,:) - popdecs(r2,:);
    
    % Combine vectors with weights
    mutants = popdecs + F(:, ones(1,D)) .* (w_feas(:, ones(1,D)).*v_feas + ...
              w_elite(:, ones(1,D)).*v_elite + w_div(:, ones(1,D)).*v_div);
    
    % Crossover with adaptive CR
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.1 * (1 - norm_cv(top_idx, ones(1,D))) .* (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
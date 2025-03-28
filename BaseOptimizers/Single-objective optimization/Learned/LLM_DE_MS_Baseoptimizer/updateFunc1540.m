% MATLAB Code
function [offspring] = updateFunc1540(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Calculate adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_cons = max(abs_cons);
    w_c = abs_cons / (max_cons + eps);
    
    % Combined weights (more emphasis on constraints)
    w = 0.4 * w_f + 0.6 * w_c;
    
    % 2. Select elite pool (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.3 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    
    % 3. Generate adaptive parameters
    F = 0.1 + 0.6 * (1 - w);  % Scaling factor
    sigma = 0.1 * w;           % Perturbation strength
    CR = 0.2 + 0.7 * w;        % Base crossover rate
    
    % Increase CR for infeasible solutions
    CR(cons > 0) = min(CR(cons > 0) + 0.3, 0.9);
    
    % 4. Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % 5. Select elite for each individual
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    
    % 6. Compute mutation vectors
    elite_dir = elite - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    perturbation = sigma .* randn(NP, 1);
    
    % 7. Mutation with elite guidance and perturbation
    mutants = popdecs + F .* elite_dir + F .* diff_dir;
    mutants = mutants + perturbation .* randn(NP, D);
    
    % 8. Crossover with adaptive CR
    mask_cr = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 9. Boundary handling with reflection and random reinitialization
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    
    % Reflection for most cases
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Random reinitialization for extreme violations (10% chance)
    extreme_mask = rand(NP, D) < 0.1 & (mask_lb | mask_ub);
    offspring(extreme_mask) = lb(extreme_mask) + (ub(extreme_mask)-lb(extreme_mask)).*rand(sum(extreme_mask(:)), 1);
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 10. Local refinement for top solutions
    top_N = max(1, round(0.2*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1534(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Weight calculation
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    min_cons = min(cons);
    max_cons = max(cons);
    w_c = (cons - min_cons) / (max_cons - min_cons + eps);
    
    w = 0.4 * w_f + 0.6 * w_c;
    
    % 2. Elite selection
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.2 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    
    % 3. Mutation
    F = 0.3 + 0.5 * (1 - w);
    sigma = 0.2;
    
    % Generate distinct random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1) + 1;
    
    % Mutation components
    elite_dir = elite - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    noise = sigma * (1 - w) .* randn(NP, D);
    
    mutants = popdecs + F .* elite_dir + F .* rand_dir + noise;
    
    % 4. Crossover
    CR = 0.1 + 0.8 * (1 - w);
    mask_cr = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 5. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % Additional local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.05 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1569(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % 1. Calculate weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    min_cons = min(abs_cons);
    max_cons = max(abs_cons);
    w_c = (abs_cons - min_cons) / (max_cons - min_cons + eps);
    
    w = (w_f + (1 - w_c)) / 2;
    
    % 2. Elite selection
    [~, sorted_idx] = sort(popfits);
    elite_num = max(1, round(0.2*NP));
    elite_idx = sorted_idx(1:elite_num);
    x_elite = mean(popdecs(elite_idx, :), 1);
    
    % 3. Best solution
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % 4. Generate random indices
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        selected = available(randperm(length(available), 2));
        r1(i) = selected(1);
        r2(i) = selected(2);
    end
    
    % 5. Calculate direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    d_best = bsxfun(@minus, x_best, popdecs);
    
    % 6. Adaptive scaling factors
    F0 = 0.5;
    F = F0 * (1 + 0.5 * sin(pi * w));
    
    % 7. Mutation
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + bsxfun(@times, F, d_elite + d_best) + ...
              bsxfun(@times, F, diff_vec);
    
    % 8. Crossover
    CR = 0.9 * w + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
    
    % Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.01 * (ub - lb) .* (1 - w(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
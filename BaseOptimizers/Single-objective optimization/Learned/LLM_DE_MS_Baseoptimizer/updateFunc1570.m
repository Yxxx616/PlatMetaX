% MATLAB Code
function [offspring] = updateFunc1570(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % 1. Calculate adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    min_cons = min(abs_cons);
    max_cons = max(abs_cons);
    w_c = (abs_cons - min_cons) / (max_cons - min_cons + eps);
    
    alpha = 0.6; % Balance between fitness and constraints
    w = alpha * w_f + (1-alpha) * (1 - w_c);
    
    % 2. Elite guidance
    [~, sorted_idx] = sort(popfits);
    elite_num = max(1, round(0.2*NP));
    elite_idx = sorted_idx(1:elite_num);
    x_elite = mean(popdecs(elite_idx, :), 1);
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = all_indices(all_indices ~= i);
        perm_idx = randperm(length(available), 2);
        r1(i) = available(perm_idx(1));
        r2(i) = available(perm_idx(2));
    end
    
    % 4. Adaptive scaling factors
    F0 = 0.5;
    F = F0 * (1 + 0.5 * sin(pi * w));
    
    % 5. Mutation with elite guidance
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + bsxfun(@times, F, d_elite) + ...
              bsxfun(@times, F, diff_vec);
    
    % 6. Crossover with adaptive CR
    CR = 0.7 + 0.3 * w;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
    
    % 8. Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.02 * (ub - lb) .* (1 - w(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc910(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select best individual considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx,:);
    end
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP,5);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 5));
    end
    
    % 3. Compute adaptive parameters
    f_avg = mean(popfits);
    f_std = std(popfits) + eps;
    phi_max = max(abs(cons)) + eps;
    
    % Fitness weights (normalized sigmoid)
    alpha = 1 ./ (1 + exp((popfits - f_avg)/f_std));
    
    % Constraint weights (tanh normalization)
    beta = tanh(abs(cons)/phi_max);
    omega = 0.5 * (1 + tanh(cons/phi_max));
    
    % Diversity weights (complementary)
    gamma = 1 - alpha - beta;
    gamma(gamma < 0) = 0;
    
    % 4. Compute mutation components
    % Elite-guided direction (F1 = 0.9)
    v1 = x_best + 0.9 * (popdecs(r(:,1),:) - popdecs(r(:,2),:));
    
    % Constraint-aware direction (F2 = 1.0)
    v2 = popdecs(r(:,3),:) + 1.0 * omega.*(popdecs(r(:,4),:) - popdecs(r(:,5),:));
    
    % Diversity direction (Gaussian around centroid)
    centroid = mean(popdecs, 1);
    sigma = 0.1 * (ub - lb);
    v3 = centroid + sigma .* randn(NP,D);
    
    % 5. Composite mutation (weighted combination)
    mutants = alpha.*v1 + beta.*v2 + gamma.*v3;
    
    % 6. Rank-based adaptive CR
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.5 + 0.4 * (ranks/NP);
    
    % 7. Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflect back into bounds if violated
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final clamping to ensure within bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
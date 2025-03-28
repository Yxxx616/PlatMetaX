% MATLAB Code
function [offspring] = updateFunc1678(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    max_cons = max(abs_cons);
    min_fit = min(popfits);
    max_fit = max(popfits);
    
    norm_cons = abs_cons ./ (max_cons + eps);
    norm_fits = (popfits - min_fit) ./ (max_fit - min_fit + eps);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    
    % Find best solution (prioritize feasible)
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask,:);
        x_best = x_best(best_idx,:);
    else
        [~, best_idx] = min(popfits + norm_cons*100);
        x_best = popdecs(best_idx,:);
    end
    
    % Compute ranks (1 is best)
    [~, rank_order] = sort(popfits + norm_cons*100);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-compute random indices
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:3);
    
    % Compute scaling factors
    F1 = 0.3 * ones(NP,1);
    F1(~feasible_mask) = 0.5 * (1 + norm_cons(~feasible_mask));
    F2 = 0.3 * (1 - norm_fits);
    
    % Compute perturbation strength
    sigma = 0.2 * (1 + norm_ranks);
    
    for i = 1:NP
        % Select random individuals
        r1 = rand_idx(i,1); 
        r2 = rand_idx(i,2);
        
        % Mutation
        mutation = popdecs(i,:) + ...
                  F1(i) * (x_best - popdecs(i,:)) + ...
                  F2(i) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                  sigma(i) * randn(1, D);
        
        % Adaptive crossover
        CR = 0.85 * (1 - norm_cons(i)) + 0.15 * norm_fits(i);
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
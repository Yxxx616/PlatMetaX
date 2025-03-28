% MATLAB Code
function [offspring] = updateFunc1680(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints (0 to 1)
    pos_cons = max(0, cons);
    norm_cons = pos_cons ./ (max(pos_cons) + eps);
    
    % Normalize fitness (0 to 1, 0 is best)
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Find best solution considering constraints
    [~, best_idx] = min(popfits + norm_cons*1e6);
    x_best = popdecs(best_idx,:);
    
    % Compute ranks (1 is best)
    [~, rank_order] = sort(popfits + norm_cons*1e6);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-compute random indices
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:3);
    
    % Compute adaptive parameters
    F1 = 0.5 * (1 + norm_cons);
    F2 = 0.3 * (1 - norm_fits);
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
        CR = 0.8 * (1 - norm_cons(i)) + 0.2 * norm_fits(i);
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with random reinitialization
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    offspring(viol_low) = lb(viol_low) + rand(sum(viol_low(:)),1) .* (ub(viol_low) - lb(viol_low));
    offspring(viol_high) = lb(viol_high) + rand(sum(viol_high(:)),1) .* (ub(viol_high) - lb(viol_high));
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
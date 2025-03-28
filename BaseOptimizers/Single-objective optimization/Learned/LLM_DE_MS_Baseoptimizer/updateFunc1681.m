% MATLAB Code
function [offspring] = updateFunc1681(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints (0 to 1)
    pos_cons = max(0, cons);
    norm_cons = pos_cons ./ (max(pos_cons) + eps);
    
    % Normalize fitness (0 to 1, 0 is best)
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Find best solution considering both fitness and constraints
    combined_score = popfits + norm_cons*1e6;
    [~, best_idx] = min(combined_score);
    x_best = popdecs(best_idx,:);
    
    % Compute ranks (1 is best)
    [~, rank_order] = sort(combined_score);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-compute random indices with fitness-based selection probability
    selection_probs = 1 - norm_ranks;
    selection_probs = selection_probs / sum(selection_probs);
    cum_probs = cumsum(selection_probs);
    
    % Generate all random numbers at once for efficiency
    rand_vals = rand(NP, 3);
    
    for i = 1:NP
        % Select individuals based on fitness rank probability
        r1 = find(rand_vals(i,1) <= cum_probs, 1);
        r2 = find(rand_vals(i,2) <= cum_probs, 1);
        while r2 == r1
            r2 = find(rand() <= cum_probs, 1);
        end
        
        % Adaptive parameters
        F1 = 0.7 * (1 - norm_cons(i));
        F2 = 0.5 * norm_fits(i);
        sigma = 0.3 * (1 + norm_cons(i));
        
        % Mutation
        mutation = popdecs(i,:) + ...
                  F1 * (x_best - popdecs(i,:)) + ...
                  F2 * (popdecs(r1,:) - popdecs(r2,:)) + ...
                  sigma * randn(1, D);
        
        % Adaptive crossover
        CR = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
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
    
    % Final bounds check and random reinitialization if needed
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = lb(viol_low) + rand(sum(viol_low(:)),1) .* (ub(viol_low) - lb(viol_low));
    offspring(viol_high) = lb(viol_high) + rand(sum(viol_high(:)),1) .* (ub(viol_high) - lb(viol_high));
    
    % Ensure all solutions are within bounds
    offspring = max(min(offspring, ub), lb);
end
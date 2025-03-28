% MATLAB Code
function [offspring] = updateFunc1683(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints (0 to 1)
    pos_cons = max(0, cons);
    norm_cons = pos_cons ./ (max(pos_cons) + eps);
    
    % Normalize fitness (0 to 1, 0 is best)
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Find best solution considering constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        x_best = popdecs(temp(best_idx),:);
    else
        [~, best_idx] = min(popfits + norm_cons*1e6);
        x_best = popdecs(best_idx,:);
    end
    
    % Rank-based selection probabilities
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    selection_probs = 1./(ranks + eps);
    selection_probs = selection_probs / sum(selection_probs);
    cum_probs = cumsum(selection_probs);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    rand_vals = rand(NP, 4);
    rand_nums = rand(NP, D);
    
    for i = 1:NP
        % Select indices using rank-based probabilities
        r1 = find(rand_vals(i,1) <= cum_probs, 1);
        r2 = find(rand_vals(i,2) <= cum_probs, 1);
        while r2 == r1
            r2 = find(rand() <= cum_probs, 1);
        end
        
        % Select third index different from i, r1, r2
        available = setdiff(1:NP, [i, r1, r2]);
        r3 = available(randi(length(available)));
        
        % Adaptive parameters
        F_i = 0.5 * (1 - norm_cons(i)) + 0.3 * norm_fits(i);
        sigma = 0.2 * norm_cons(i);
        
        % Mutation
        mutation = x_best + ...
                  F_i * (popdecs(r1,:) - popdecs(r2,:)) + ...
                  sigma * (popdecs(r3,:) - popdecs(i,:));
        
        % Adaptive crossover
        CR_i = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        j_rand = randi(D);
        mask = rand_nums(i,:) < CR_i;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Enhanced boundary handling
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % Reflection with adaptive factor
    reflect_factor = 1 + rand(size(offspring));
    offspring(viol_low) = lb(viol_low) - reflect_factor(viol_low) .* (offspring(viol_low) - lb(viol_low));
    offspring(viol_high) = ub(viol_high) + reflect_factor(viol_high) .* (ub(viol_high) - offspring(viol_high));
    
    % Random reinitialization if still out of bounds
    still_viol = (offspring < lb) | (offspring > ub);
    offspring(still_viol) = lb(still_viol) + rand(sum(still_viol(:)), 1) .* (ub(still_viol) - lb(still_viol));
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
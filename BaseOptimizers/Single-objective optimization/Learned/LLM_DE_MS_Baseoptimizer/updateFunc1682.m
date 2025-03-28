% MATLAB Code
function [offspring] = updateFunc1682(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints (0 to 1)
    pos_cons = max(0, cons);
    norm_cons = pos_cons ./ (max(pos_cons) + eps);
    
    % Normalize fitness (0 to 1, 0 is best)
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Find best feasible solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        x_best = popdecs(temp(best_idx),:);
    else
        [~, best_idx] = min(popfits + norm_cons*1e6);
        x_best = popdecs(best_idx,:);
    end
    
    % Compute selection probabilities based on fitness
    [~, sorted_idx] = sort(popfits);
    inv_rank = zeros(NP,1);
    inv_rank(sorted_idx) = NP:-1:1;
    selection_probs = inv_rank / sum(inv_rank);
    cum_probs = cumsum(selection_probs);
    
    % Initialize offspring and pre-generate random numbers
    offspring = zeros(NP, D);
    rand_vals = rand(NP, 5);
    rand_nums = rand(NP, D);
    
    for i = 1:NP
        % Select indices using fitness-based probabilities
        r1 = find(rand_vals(i,1) <= cum_probs, 1);
        r2 = find(rand_vals(i,2) <= cum_probs, 1);
        while r2 == r1
            r2 = find(rand() <= cum_probs, 1);
        end
        
        % Select two additional random indices
        available = setdiff(1:NP, [i, r1, r2]);
        r3 = available(randi(length(available)));
        available = setdiff(available, r3);
        r4 = available(randi(length(available)));
        
        % Determine base vector
        if norm_cons(i) < 0.5
            x_base = x_best;
        else
            x_base = popdecs(i,:);
        end
        
        % Compute adaptive parameters
        F1 = 0.5 * (1 - norm_cons(i));
        F2 = 0.3 * norm_fits(i);
        sigma = 0.2 * norm_cons(i);
        
        % Mutation
        mutation = x_base + ...
                  F1 * (x_best - popdecs(i,:)) + ...
                  F2 * (popdecs(r1,:) - popdecs(r2,:)) + ...
                  sigma * (popdecs(r3,:) - popdecs(r4,:));
        
        % Adaptive crossover
        CR = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        j_rand = randi(D);
        mask = rand_nums(i,:) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with reflection and reinitialization
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % Reflection
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Random reinitialization if still out of bounds
    still_viol = (offspring < lb) | (offspring > ub);
    rand_vals = rand(sum(still_viol(:)), 1);
    offspring(still_viol) = lb(still_viol) + rand_vals .* (ub(still_viol) - lb(still_viol));
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc1664(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps);
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Identify feasible and infeasible solutions
    feasible_mask = cons <= 0;
    feasible_pop = popdecs(feasible_mask, :);
    num_feasible = size(feasible_pop, 1);
    
    % Find best solution (prioritize feasible solutions)
    if num_feasible > 0
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = feasible_pop(best_idx, :);
    else
        [~, best_idx] = min(popfits + 100*norm_cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Select random indices
        candidates = 1:NP;
        candidates(i) = [];
        r_idx = candidates(randperm(length(candidates), 4));
        r1 = r_idx(1); r2 = r_idx(2); r3 = r_idx(3); r4 = r_idx(4);
        
        % Select feasible and infeasible solutions if available
        if num_feasible > 0
            feasible_idx = randi(num_feasible);
            x_feas = feasible_pop(feasible_idx, :);
        else
            x_feas = popdecs(r3, :);
        end
        x_infeas = popdecs(r4, :);
        
        % Adaptive parameters
        F1 = 0.8 * (1 - norm_cons(i));
        F2 = 0.6 * norm_fits(i);
        F3 = 0.4 * norm_cons(i);
        CR = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        
        % Composite mutation
        mutation = popdecs(i,:) + ...
                  F1 .* (x_best - popdecs(i,:)) + ...
                  F2 .* (popdecs(r1,:) - popdecs(r2,:)) + ...
                  F3 .* (x_feas - x_infeas);
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with reflection and randomization
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    extreme_viol = viol_low | viol_high;
    
    % Reflection for moderate violations
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Random reinitialization for extreme violations
    extreme_mask = extreme_viol & (rand(NP,D) < 0.3);
    offspring(extreme_mask) = lb(extreme_mask) + rand(sum(extreme_mask(:)),1).*...
                            (ub(extreme_mask)-lb(extreme_mask));
    
    % Ensure final solutions are within bounds
    offspring = max(min(offspring, ub), lb);
end
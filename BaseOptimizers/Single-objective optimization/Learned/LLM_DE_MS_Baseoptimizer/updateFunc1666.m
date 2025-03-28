% MATLAB Code
function [offspring] = updateFunc1666(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps);
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    feasible_pop = popdecs(feasible_mask, :);
    num_feasible = sum(feasible_mask);
    
    % Find best solution (prioritize feasible)
    if num_feasible > 0
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = feasible_pop(best_idx, :);
    else
        [~, best_idx] = min(popfits + norm_cons*100);
        x_best = popdecs(best_idx, :);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-compute random indices
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:end); % Exclude self
    
    for i = 1:NP
        % Select random individuals
        r1 = rand_idx(i,1); r2 = rand_idx(i,2);
        
        % Select feasible and infeasible solutions
        if num_feasible > 0
            feas_idx = randi(num_feasible);
            x_feas = feasible_pop(feas_idx, :);
        else
            x_feas = popdecs(r1, :);
        end
        
        % Select an infeasible solution
        infeas_mask = ~feasible_mask;
        if any(infeas_mask)
            infeas_idx = find(infeas_mask);
            sel_idx = randi(length(infeas_idx));
            x_infeas = popdecs(infeas_idx(sel_idx), :);
        else
            x_infeas = popdecs(r2, :);
        end
        
        % Adaptive parameters
        F1 = 0.8 * (1 - norm_cons(i));
        F2 = 0.5 * norm_cons(i);
        F3 = 0.3 * norm_fits(i);
        CR = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        
        % Directional mutation
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
    
    % Boundary handling with adaptive reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % Reflection with adaptive probability
    reflect_prob = 0.5 * (1 - norm_cons);
    reflect_mask = rand(NP,D) < repmat(reflect_prob, 1, D);
    
    offspring(viol_low & reflect_mask) = 2*lb(viol_low & reflect_mask) - offspring(viol_low & reflect_mask);
    offspring(viol_high & reflect_mask) = 2*ub(viol_high & reflect_mask) - offspring(viol_high & reflect_mask);
    
    % Random reinitialization for remaining violations
    rand_mask = (viol_low | viol_high) & ~reflect_mask;
    offspring(rand_mask) = lb(rand_mask) + rand(sum(rand_mask(:)),1) .* (ub(rand_mask) - lb(rand_mask));
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
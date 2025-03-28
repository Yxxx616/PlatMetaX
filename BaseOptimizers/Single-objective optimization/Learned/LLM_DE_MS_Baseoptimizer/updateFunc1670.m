% MATLAB Code
function [offspring] = updateFunc1670(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps;
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    feasible_pop = popdecs(feasible_mask, :);
    feasible_fits = popfits(feasible_mask);
    num_feasible = sum(feasible_mask);
    
    % Find best solution (prioritize feasible)
    if num_feasible > 0
        [~, best_idx] = min(feasible_fits);
        x_best = feasible_pop(best_idx, :);
    else
        [~, best_idx] = min(popfits + norm_cons*100);
        x_best = popdecs(best_idx, :);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-compute random indices
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:3);
    
    % Compute feasibility weights if any feasible solutions exist
    if num_feasible > 0
        feas_weights = 1 ./ (1 + abs(cons(feasible_mask)));
        feas_weights = feas_weights ./ sum(feas_weights);
    end
    
    for i = 1:NP
        % Select random individuals
        r1 = rand_idx(i,1); 
        r2 = rand_idx(i,2);
        
        % Adaptive parameters
        F1 = 0.8 * (1 - norm_cons(i));
        F2 = 0.5 * norm_cons(i);
        F3 = 0.3 * norm_fits(i);
        
        % Constraint-guided component
        if num_feasible > 0
            % Weighted average of feasible solutions
            feas_component = zeros(1, D);
            for k = 1:num_feasible
                feas_component = feas_component + feas_weights(k) * (feasible_pop(k,:) - popdecs(i,:));
            end
        else
            feas_component = popdecs(r1,:) - popdecs(r2,:);
        end
        
        % Enhanced mutation
        mutation = popdecs(i,:) + ...
                  F1 .* (x_best - popdecs(i,:)) + ...
                  F2 .* (popdecs(r1,:) - popdecs(r2,:)) + ...
                  F3 .* feas_component;
        
        % Adaptive crossover
        CR = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with adaptive reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % Reflection probability based on solution quality
    reflect_prob = 0.5 * (1 - norm_cons) + 0.5 * (1 - norm_fits);
    reflect_mask = rand(NP,D) < repmat(reflect_prob, 1, D);
    
    % Apply reflection
    offspring(viol_low & reflect_mask) = 2*lb(viol_low & reflect_mask) - offspring(viol_low & reflect_mask);
    offspring(viol_high & reflect_mask) = 2*ub(viol_high & reflect_mask) - offspring(viol_high & reflect_mask);
    
    % Random reinitialization for remaining violations
    rand_mask = (viol_low | viol_high) & ~reflect_mask;
    offspring(rand_mask) = lb(rand_mask) + rand(sum(rand_mask(:)),1) .* (ub(rand_mask) - lb(rand_mask));
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc886(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility-aware best selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_best = popdecs(temp(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Normalize constraints and fitness
    c_max = max(abs(cons));
    norm_c = abs(cons) / (c_max + 1e-12);
    f_min = min(popfits);
    f_max = max(popfits);
    norm_f = (popfits - f_min) / (f_max - f_min + 1e-12);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Adaptive parameters
        F_i = 0.5 * (1 + tanh(1 - norm_c(i)));
        CR_i = 0.9 * (1 - norm_f(i));
        
        % Select three distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Hybrid mutation
        diff1 = popdecs(r1,:) - popdecs(r2,:);
        diff2 = x_best - popdecs(r3,:);
        mutant = x_best + F_i * diff1 + F_i * diff2;
        
        % Exponential crossover
        j_rand = randi(D);
        L = 0;
        while (rand() <= CR_i) && (L < D)
            L = L + 1;
        end
        indices = mod((j_rand:j_rand+L-1)-1, D) + 1;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,indices) = mutant(indices);
    end
    
    % Boundary handling - midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = (lb_rep(below) + offspring(below)) / 2;
    offspring(above) = (ub_rep(above) + offspring(above)) / 2;
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    offspring(best_idx,:) = x_best;
end
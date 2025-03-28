% MATLAB Code
function [offspring] = updateFunc885(popdecs, popfits, cons)
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
    
    % Normalize constraints [0,1]
    c_max = max(abs(cons));
    norm_c = abs(cons) / (c_max + 1e-12);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Adaptive parameters
        F_i = 0.4 + 0.6 * exp(-5 * norm_c(i));
        CR_i = 0.1 + 0.8 * (1 - tanh(3 * norm_c(i)));
        
        % Select three distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Hybrid mutation
        diff1 = popdecs(r1,:) - popdecs(i,:);
        diff2 = popdecs(r2,:) - popdecs(r3,:);
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
    
    % Boundary handling - reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    offspring(best_idx,:) = x_best;
end
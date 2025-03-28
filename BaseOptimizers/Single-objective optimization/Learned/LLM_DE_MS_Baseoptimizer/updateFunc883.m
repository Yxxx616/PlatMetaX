% MATLAB Code
function [offspring] = updateFunc883(popdecs, popfits, cons)
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
    c_min = min(cons);
    c_max = max(cons);
    norm_c = (cons - c_min) / (c_max - c_min + 1e-12);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Adaptive parameters
        F_i = 0.4 + 0.3 * norm_c(i) + 0.3 * randn();
        CR_i = 0.1 + 0.8 * (1 - norm_c(i));
        
        % Select six distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 6));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); 
        r4 = idx(4); r5 = idx(5); r6 = idx(6);
        
        % Direction vectors
        exploit_dir = x_best - popdecs(i,:);
        explore_dir = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
        perturbation = 0.2 * (popdecs(r5,:) - popdecs(r6,:));
        mutant = popdecs(i,:) + F_i * ((1-norm_c(i))*exploit_dir + norm_c(i)*explore_dir) + perturbation;
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR_i;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling - midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = (lb_rep(below) + popdecs(below)) / 2;
    offspring(above) = (ub_rep(above) + popdecs(above)) / 2;
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    offspring(best_idx,:) = x_best;
end
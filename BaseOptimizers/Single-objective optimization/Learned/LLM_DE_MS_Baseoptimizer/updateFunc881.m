% MATLAB Code
function [offspring] = updateFunc881(popdecs, popfits, cons)
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
    
    % Constraint weight (sigmoid)
    w = 1 ./ (1 + exp(-5 * norm_c));
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Adaptive parameters
        F_i = 0.5 * (1 + tanh(5 * (0.5 - w(i)))) * (1 + 0.2 * randn());
        CR_i = 0.9 * (1 - w(i)) + 0.1;
        
        % Select three distinct random vectors (excluding current and best)
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Direction vectors
        exploit_dir = x_best - popdecs(i,:);
        explore_dir = popdecs(r1,:) - popdecs(r2,:);
        direction = (1-w(i))*exploit_dir + w(i)*explore_dir;
        
        % Mutation with additional perturbation
        if i ~= best_idx
            perturbation = 0.1 * (x_best - popdecs(r3,:));
        else
            perturbation = 0;
        end
        mutant = popdecs(i,:) + F_i * direction + perturbation;
        
        % Binomial crossover
        mask = rand(1,D) < CR_i;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
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
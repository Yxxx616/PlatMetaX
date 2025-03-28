% MATLAB Code
function [offspring] = updateFunc882(popdecs, popfits, cons)
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
        F_i = 0.5 + 0.3 * tanh(5 * (0.5 - w(i))) + 0.2 * randn();
        CR_i = 0.9 * (1 - w(i)) + 0.1;
        
        % Select five distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 5));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4); r5 = idx(5);
        
        % Direction vectors
        exploit_dir = x_best - popdecs(i,:);
        explore_dir = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
        direction = (1-w(i))*exploit_dir + w(i)*explore_dir;
        
        % Mutation with additional perturbation
        perturbation = 0.1 * (x_best - popdecs(r5,:));
        mutant = popdecs(i,:) + F_i * direction + perturbation;
        
        % Exponential crossover
        j_rand = randi(D);
        L = floor(CR_i^D * D);
        mask = false(1,D);
        for k = 0:L-1
            idx = mod(j_rand + k - 1, D) + 1;
            mask(idx) = true;
        end
        
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
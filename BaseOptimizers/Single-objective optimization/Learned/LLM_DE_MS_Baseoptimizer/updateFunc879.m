% MATLAB Code
function [offspring] = updateFunc879(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best solution considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Normalize constraints [0,1]
    c_min = min(cons);
    c_max = max(cons);
    norm_c = (cons - c_min) / (c_max - c_min + 1e-12);
    
    % Constraint weight (sigmoid)
    k = 5; % steepness parameter
    w = 1 ./ (1 + exp(-k * norm_c));
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Adaptive parameters
        F_i = 0.5 + 0.3 * randn(); % Normal distribution around 0.5
        CR_i = 0.5 + 0.4 * (1 - w(i)); % CR ∈ [0.5,0.9]
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Constraint-aware mutation
        direction = (1-w(i))*(x_best - popdecs(i,:)) + w(i)*(popdecs(r1,:) - popdecs(r2,:));
        mutant = popdecs(i,:) + F_i * direction;
        
        % Binomial crossover
        mask = rand(1,D) < CR_i;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling - reflection for all
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Clamp any remaining out of bounds (shouldn't happen with reflection)
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    offspring(best_idx,:) = x_best;
end
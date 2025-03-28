% MATLAB Code
function [offspring] = updateFunc884(popdecs, popfits, cons)
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
        F_i = 0.5 * (1 + tanh(2 * norm_c(i) - 1));
        CR_i = 0.9 * (1 - exp(-5 * norm_c(i)));
        
        % Select four distinct random vectors
        candidates = setdiff(1:NP, [i, best_idx]);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Hybrid mutation
        base = popdecs(i,:);
        best_diff = x_best - base;
        rand_diff1 = popdecs(r1,:) - popdecs(r2,:);
        rand_diff2 = popdecs(r3,:) - popdecs(r4,:);
        
        mutant = base + F_i * best_diff + F_i * rand_diff1 + 0.5 * F_i * rand_diff2;
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR_i;
        mask(j_rand) = true;
        
        offspring(i,:) = base;
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
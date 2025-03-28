% MATLAB Code
function [offspring] = updateFunc1610(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 1e6*abs(cons));
        elite = popdecs(elite_idx,:);
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.5 + 0.5 * (popfits - f_min) ./ (f_max - f_min + eps);
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    C = 0.2 * (1 + c_abs ./ (c_max + eps));
    
    % 3. Generate offspring
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Mutation with elite guidance and constraint-aware perturbation
        mutation = elite + F(i) * (popdecs(r1,:) - popdecs(r2,:)) ...
                 + C(i) * randn(1, D) .* (popdecs(r3,:) - popdecs(r4,:));
        
        % Binomial crossover with adaptive CR
        CR = 0.9 - 0.5 * (c_abs(i) / (c_max + eps));
        mask = rand(1, D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Boundary handling with reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
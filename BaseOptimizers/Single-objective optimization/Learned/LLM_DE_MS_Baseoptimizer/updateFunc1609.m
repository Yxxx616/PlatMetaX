% MATLAB Code
function [offspring] = updateFunc1609(popdecs, popfits, cons)
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
    
    % Fitness-based weights
    w = exp(-popfits);
    w = w ./ (sum(w) + eps);
    
    % Constraint-aware scaling
    c_abs = abs(cons);
    c_max = max(c_abs);
    C = 0.1 * (1 + c_abs ./ (c_max + eps));
    
    % 3. Generate mutation vectors
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Compute weighted difference vector
        weighted_diff = w(r3) * popdecs(r3,:) - w(r4) * popdecs(r4,:);
        
        % Mutation with elite guidance
        mutation = elite + F(i) * (popdecs(r1,:) - popdecs(r2,:)) + weighted_diff;
        
        % Add constraint-aware perturbation
        mutation = mutation + C(i) * randn(1, D);
        
        % Binomial crossover
        CR = 0.9 - 0.4 * (c_abs(i) / (c_max + eps));
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
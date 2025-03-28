% MATLAB Code
function [offspring] = updateFunc1611(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution considering both fitness and constraints
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    % 2. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.4 + 0.4 * (popfits - f_min) ./ (f_max - f_min + eps);
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    C = 0.3 * (1 + c_abs ./ (c_max + eps));
    
    % 3. Generate offspring
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Enhanced mutation with elite guidance and constraint-aware perturbation
        mutation = elite + F(i) * (popdecs(r1,:) - popdecs(r2,:)) ...
                 + C(i) * (popdecs(r3,:) - popdecs(r4,:));
        
        % Adaptive crossover considering constraints
        CR = 0.8 - 0.4 * (c_abs(i) / (c_max + eps));
        mask = rand(1, D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Boundary handling with random reinitialization
    out_of_bounds = (offspring < lb) | (offspring > ub);
    for j = 1:D
        if any(out_of_bounds(:,j))
            offspring(out_of_bounds(:,j),j) = lb(j) + (ub(j)-lb(j)) * rand(sum(out_of_bounds(:,j)),1);
        end
    end
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
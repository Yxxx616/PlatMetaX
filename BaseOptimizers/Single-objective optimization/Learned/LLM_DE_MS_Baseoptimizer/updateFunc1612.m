% MATLAB Code
function [offspring] = updateFunc1612(popdecs, popfits, cons)
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
    f_range = f_max - f_min + eps;
    F = 0.5 * (1 + (popfits - f_min) ./ f_range);
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    C = 0.3 * (1 + c_abs ./ (c_max + eps));
    
    D = 0.2 * (1 - (popfits - f_min) ./ f_range);
    
    % 3. Generate offspring
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select 5 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 5));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4); r5 = idx(5);
        
        % Enhanced mutation with multiple components
        mutation = elite + F(i) * (popdecs(r1,:) - popdecs(r2,:)) ...
                 + C(i) * (popdecs(r3,:) - popdecs(r4,:)) ...
                 + D(i) * (popdecs(r5,:) - elite);
        
        % Adaptive crossover considering constraints
        CR = 0.9 - 0.5 * (c_abs(i) / (c_max + eps));
        mask = rand(1, D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Boundary handling with reflection
    out_of_bounds = (offspring < lb) | (offspring > ub);
    for j = 1:D
        if any(out_of_bounds(:,j))
            % Reflect back into bounds
            temp = offspring(out_of_bounds(:,j),j);
            temp = 2*lb(j) - temp .* (temp < lb(j)) + 2*ub(j) - temp .* (temp > ub(j));
            % Randomize if still out of bounds
            still_out = (temp < lb(j)) | (temp > ub(j));
            temp(still_out) = lb(j) + (ub(j)-lb(j)) * rand(sum(still_out),1);
            offspring(out_of_bounds(:,j),j) = temp;
        end
    end
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
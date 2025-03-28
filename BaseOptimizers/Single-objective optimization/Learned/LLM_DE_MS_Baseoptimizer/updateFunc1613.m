% MATLAB Code
function [offspring] = updateFunc1613(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution considering both fitness and constraints
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    % 2. Compute adaptive parameters based on fitness ranks
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    F = 0.4 + 0.5 * (ranks ./ NP);
    
    % 3. Constraint-aware parameters
    c_abs = abs(cons);
    c_max = max(c_abs);
    CR = 0.9 - 0.5 * (c_abs ./ (c_max + eps));
    
    % 4. Generate offspring
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Enhanced mutation with elite guidance
        mutation = elite + F(i) * (popdecs(r1,:) - popdecs(r2,:)) ...
                 + 0.3 * (popdecs(r3,:) - popdecs(r4,:));
        
        % Constraint-aware crossover
        mask = rand(1, D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 5. Boundary handling with adaptive reflection
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
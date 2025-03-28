% MATLAB Code
function [offspring] = updateFunc1616(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool (top 30% considering constraints)
    penalty = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(penalty);
    elite_size = ceil(NP*0.3);
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    F1 = 0.5 + 0.3 * (ranks ./ NP);
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    F2 = 0.2 * (1 - c_abs ./ (c_max + eps));
    CR = 0.9 - 0.5 * (c_abs ./ (c_max + eps));
    
    % 3. Generate offspring
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite from pool
        elite = elite_pool(randi(elite_size), :);
        
        % Select 4 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Enhanced mutation with three components
        mutation = elite + F1(i) * (popdecs(r1,:) - popdecs(r2,:)) ...
                 + F2(i) * (popdecs(r3,:) - popdecs(r4,:)) .* (1 + c_abs(i)/(c_max + eps)) ...
                 + 0.1 * (1 - ranks(i)/NP) * randn(1, D);
        
        % Adaptive crossover
        mask = rand(1, D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Boundary handling with reflection
    out_of_bounds = (offspring < lb) | (offspring > ub);
    for j = 1:D
        if any(out_of_bounds(:,j))
            viol_low = offspring(:,j) < lb(j);
            viol_high = offspring(:,j) > ub(j);
            offspring(viol_low,j) = 2*lb(j) - offspring(viol_low,j);
            offspring(viol_high,j) = 2*ub(j) - offspring(viol_high,j);
            
            % Final randomization if still out of bounds
            still_viol = (offspring(:,j) < lb(j)) | (offspring(:,j) > ub(j));
            offspring(still_viol,j) = lb(j) + (ub(j)-lb(j)) * rand(sum(still_viol),1);
        end
    end
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub), lb);
end
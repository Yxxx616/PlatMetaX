% MATLAB Code
function [offspring] = updateFunc1615(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite pool (top 20%)
    penalty = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(penalty);
    elite_pool = popdecs(sorted_idx(1:ceil(NP*0.2)), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    F_base = 0.5 + 0.3 * (ranks ./ NP);
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    F_cons = 0.2 * (1 - c_abs ./ (c_max + eps));
    F_total = F_base + F_cons;
    CR = 0.9 - 0.5 * (c_abs ./ (c_max + eps));
    
    % 3. Generate offspring
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite from pool
        elite = elite_pool(randi(size(elite_pool,1)), :);
        
        % Select 4 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Enhanced mutation with elite guidance and constraint awareness
        mutation = elite + F_total(i) * (popdecs(r1,:) - popdecs(r2,:)) ...
                 + 0.5 * F_cons(i) * (popdecs(r3,:) - popdecs(r4,:));
        
        % Add fitness-based perturbation
        mutation = mutation + 0.1 * (1 - ranks(i)/NP) * randn(1, D);
        
        % Constraint-aware crossover
        mask = rand(1, D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Boundary handling with randomization
    out_of_bounds = (offspring < lb) | (offspring > ub);
    for j = 1:D
        if any(out_of_bounds(:,j))
            offspring(out_of_bounds(:,j),j) = lb(j) + (ub(j)-lb(j)) * rand(sum(out_of_bounds(:,j)),1);
        end
    end
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc1628(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool (top 30% considering both fitness and constraints)
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(3, ceil(NP*0.3));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    
    % Adaptive parameters
    F = 0.5 * (1 + ranks./NP) .* (1 - c_abs./c_max);
    Fc = 0.3 * c_abs./c_max;
    Fd = 0.1 * (1 - ranks./NP);
    CR = 0.9 * (1 - ranks./NP) .* (1 - c_abs./c_max);
    
    % 3. Generate offspring (vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite and random vectors
        elite = elite_pool(randi(elite_size), :);
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Composite mutation
        mutation = elite + F(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                   Fc(i)*(popdecs(r3,:)-popdecs(r4,:)) + ...
                   Fd(i)*randn(1,D);
        
        % Adaptive crossover
        mask = rand(1,D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 4. Smart boundary handling
    for j = 1:D
        % For lower bound violations
        viol_low = offspring(:,j) < lb(j);
        if any(viol_low)
            offspring(viol_low,j) = 0.5*(lb(j) + popdecs(viol_low,j));
        end
        
        % For upper bound violations
        viol_high = offspring(:,j) > ub(j);
        if any(viol_high)
            offspring(viol_high,j) = 0.5*(ub(j) + popdecs(viol_high,j));
        end
    end
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
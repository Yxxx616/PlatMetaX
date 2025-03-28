% MATLAB Code
function [offspring] = updateFunc1638(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool (top 20% or at least 2)
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(2, ceil(NP*0.2));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute ranks and constraint info
    [~, rank_order] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    
    % 3. Adaptive parameters
    F1 = 0.5 + 0.3*(1 - ranks./NP);
    F2 = 0.3*(1 - c_abs./c_max);
    F3 = 0.2*ranks./NP;
    
    % 4. Generate offspring (fully vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vector
        elite_idx = randi(elite_size);
        elite = elite_pool(elite_idx, :);
        
        % Select four distinct random vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Composite mutation
        mutation = elite + F1(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                  F2(i)*(popdecs(r3,:)-popdecs(r4,:)) + ...
                  F3(i)*randn(1,D);
        
        % Adaptive crossover
        CR = 0.9*(1 - ranks(i)/NP)*(1 - c_abs(i)/c_max);
        mask = rand(1,D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 5. Smart boundary handling
    % Reflection with probability based on constraint violation
    reflect_prob = 1 - c_abs./c_max;
    reflect_mask = rand(NP,D) < reflect_prob;
    
    viol_low = offspring < lb;
    offspring(viol_low & reflect_mask) = 2*lb(viol_low & reflect_mask) - offspring(viol_low & reflect_mask);
    offspring(viol_low & ~reflect_mask) = lb(viol_low & ~reflect_mask) + rand(sum(viol_low(:) & ~reflect_mask(:)),1).*(ub(viol_low & ~reflect_mask)-lb(viol_low & ~reflect_mask));
    
    viol_high = offspring > ub;
    offspring(viol_high & reflect_mask) = 2*ub(viol_high & reflect_mask) - offspring(viol_high & reflect_mask);
    offspring(viol_high & ~reflect_mask) = lb(viol_high & ~reflect_mask) + rand(sum(viol_high(:) & ~reflect_mask(:)),1).*(ub(viol_high & ~reflect_mask)-lb(viol_high & ~reflect_mask));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
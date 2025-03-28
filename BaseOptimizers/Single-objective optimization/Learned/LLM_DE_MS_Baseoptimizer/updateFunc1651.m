% MATLAB Code
function [offspring] = updateFunc1651(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware fitness weighting
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(abs(cons));
    weights = 1./(1 + exp(0.5*fit_rank + 0.5*cons_rank));
    weights = weights / sum(weights);
    
    % 2. Elite selection (top 20%)
    elite_size = max(2, ceil(NP*0.2));
    [~, elite_idx] = sort(popfits);
    elite_pool = popdecs(elite_idx(1:elite_size), :);
    x_best = popdecs(elite_idx(1), :);
    
    % 3. Adaptive parameters
    F_base = 0.5;
    CR_base = 0.9;
    F = F_base * (0.5 + 0.5*weights);
    CR = CR_base * (0.5 + 0.5*weights);
    
    % 4. Vectorized offspring generation
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vectors with probability proportional to weights
        elite_probs = weights(elite_idx(1:elite_size));
        elite_probs = elite_probs / sum(elite_probs);
        e_idx = randsample(elite_size, 2, true, elite_probs);
        e1 = elite_pool(e_idx(1), :);
        e2 = elite_pool(e_idx(2), :);
        
        % Select random distinct vectors
        candidates = setdiff(1:NP, i);
        r_idx = candidates(randperm(length(candidates), 6));
        
        % Composite mutation
        w_diff1 = weights(r_idx(1))/(weights(r_idx(1))+weights(r_idx(2))+eps) * ...
                 (popdecs(r_idx(1),:) - popdecs(r_idx(2),:));
        w_diff2 = weights(r_idx(3))/(weights(r_idx(3))+weights(r_idx(4))+eps) * ...
                 (popdecs(r_idx(3),:) - popdecs(r_idx(4),:));
        w_diff3 = weights(r_idx(5))/(weights(r_idx(5))+weights(r_idx(6))+eps) * ...
                 (popdecs(r_idx(5),:) - popdecs(r_idx(6),:));
        
        mutation = popdecs(i,:) + F(i) * (...
            (x_best - popdecs(i,:)) + ...
            weights(e_idx(1))*(e1 - popdecs(i,:)) + ...
            weights(e_idx(2))*(e2 - popdecs(i,:)) + ...
            0.33*(w_diff1 + w_diff2 + w_diff3));
        
        % Enhanced crossover
        mask = rand(1,D) < CR(i)*(1+weights(i));
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 5. Smart boundary handling
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = min(ub(viol_low), max(lb(viol_low), ...
        2*lb(viol_low) - offspring(viol_low)));
    offspring(viol_high) = min(ub(viol_high), max(lb(viol_high), ...
        2*ub(viol_high) - offspring(viol_high)));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
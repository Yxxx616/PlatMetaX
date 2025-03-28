% MATLAB Code
function [offspring] = updateFunc1652(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Fitness-constraint balanced weights
    alpha = 0.7; beta = 0.3;
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (abs(cons) - min(abs(cons))) / (max(abs(cons)) - min(abs(cons)) + eps);
    weights = 1./(1 + exp(-(alpha*norm_fits + beta*norm_cons)));
    weights = weights / sum(weights);
    
    % 2. Elite selection (top 30% with probability)
    elite_size = max(3, ceil(NP*0.3));
    [~, elite_idx] = sort(popfits);
    elite_pool = popdecs(elite_idx(1:elite_size), :);
    elite_weights = weights(elite_idx(1:elite_size));
    elite_weights = elite_weights / sum(elite_weights);
    x_best = popdecs(elite_idx(1), :);
    
    % 3. Adaptive parameters
    F = 0.5 * (0.5 + 0.5*weights);
    CR = 0.9 * (0.5 + 0.5*weights);
    
    % 4. Vectorized offspring generation
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vectors
        e_idx = randsample(elite_size, 2, true, elite_weights);
        e1 = elite_pool(e_idx(1), :);
        e2 = elite_pool(e_idx(2), :);
        
        % Select random distinct vectors
        candidates = setdiff(1:NP, i);
        r_idx = candidates(randperm(length(candidates), 4));
        
        % Weighted differential components
        w1 = weights(r_idx(1))/(weights(r_idx(1))+weights(r_idx(2))+eps);
        w2 = weights(r_idx(3))/(weights(r_idx(3))+weights(r_idx(4))+eps);
        delta = w1*(popdecs(r_idx(1),:) - popdecs(r_idx(2),:)) + ...
                w2*(popdecs(r_idx(3),:) - popdecs(r_idx(4),:));
        
        % Composite mutation
        mutation = popdecs(i,:) + F(i) * (...
            (x_best - popdecs(i,:)) + ...
            weights(e_idx(1))*(e1 - popdecs(i,:)) + ...
            weights(e_idx(2))*(e2 - popdecs(i,:))) + ...
            0.5 * F(i) * delta;
        
        % Smart crossover
        mask = rand(1,D) < CR(i)*weights(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 5. Boundary handling with reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = min(ub(viol_low), 2*lb(viol_low) - offspring(viol_low));
    offspring(viol_high) = max(lb(viol_high), 2*ub(viol_high) - offspring(viol_high));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
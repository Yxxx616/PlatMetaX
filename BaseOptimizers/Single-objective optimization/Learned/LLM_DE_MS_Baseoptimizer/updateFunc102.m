% MATLAB Code
function [offspring] = updateFunc102(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % 1. Weight calculation
    c_max = max(abs(cons));
    norm_cons = abs(cons) / (c_max + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    alpha = 0.7;
    weights = alpha * norm_cons + (1-alpha) * (1 - norm_fits);
    weights = weights / sum(weights);
    
    % 2. Elite selection
    beta = 0.5;
    elite_scores = popfits + beta * abs(cons);
    [~, best_idx] = min(elite_scores);
    x_best = popdecs(best_idx, :);
    
    % 3. Mutation
    F = 0.8;
    v = zeros(NP, D);
    
    for i = 1:NP
        % Select 4 distinct individuals using weights
        candidates = setdiff(1:NP, i);
        idx = datasample(candidates, 4, 'Replace', false, 'Weights', weights(candidates));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Mutation vector
        v(i,:) = x_best + F*(popdecs(r1,:)-popdecs(r2,:)) + F*(popdecs(r3,:)-popdecs(r4,:));
    end
    
    % 4. Adaptive CR
    CR = 0.1 + 0.7 * (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    % 5. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
end
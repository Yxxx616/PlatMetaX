% MATLAB Code
function [offspring] = updateFunc103(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % 1. Normalization and weight calculation
    c_abs = abs(cons);
    c_max = max(c_abs);
    norm_cons = c_abs / (c_max + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    alpha = 0.6;
    weights = alpha * norm_cons + (1-alpha) * (1 - norm_fits);
    weights = weights / sum(weights);
    
    % 2. Elite selection (top 3)
    beta = 0.5;
    elite_scores = popfits + beta * c_abs;
    [~, elite_idx] = sort(elite_scores);
    elite_idx = elite_idx(1:3);
    elites = popdecs(elite_idx, :);
    
    % 3. Adaptive mutation
    F_base = 0.8;
    v = zeros(NP, D);
    
    for i = 1:NP
        % Select base vector (weighted combination of elites)
        elite_weights = weights(elite_idx) / sum(weights(elite_idx));
        x_base = elite_weights' * elites;
        
        % Select 4 distinct individuals using weights
        candidates = setdiff(1:NP, i);
        idx = datasample(candidates, 4, 'Replace', false, 'Weights', weights(candidates));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Adaptive F parameters
        F1 = 0.8 * (1 - norm_cons(i)) + 0.2;
        F2 = 0.6 * (1 - norm_fits(i)) + 0.2;
        
        % Mutation vector
        v(i,:) = x_base + F1*(popdecs(r1,:)-popdecs(r2,:)) + F2*(popdecs(r3,:)-popdecs(r4,:));
    end
    
    % 4. Dynamic crossover rate
    CR = 0.1 + 0.8 * (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    % 5. Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % 6. Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Boundary control
    lb = -1000 * ones(1,D);
    ub = 1000 * ones(1,D);
    offspring = min(max(offspring, lb), ub);
end
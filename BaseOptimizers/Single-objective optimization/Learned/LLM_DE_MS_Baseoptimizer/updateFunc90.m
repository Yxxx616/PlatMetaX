function [offspring] = updateFunc90(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs(cons));
    if c_max == 0
        norm_cons = zeros(size(cons));
    else
        norm_cons = abs(cons) / c_max;
    end
    
    % Calculate combined weights (60% fitness, 40% constraints)
    weights = 0.6 * norm_fits + 0.4 * norm_cons;
    weights = 1 - weights; % Invert for selection probability
    weights = weights / sum(weights); % Normalize
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Parameters
    CR = 0.9;
    
    % Pre-compute cumulative probabilities for selection
    cum_probs = cumsum(weights);
    
    % Vectorized selection of r1, r2, r3
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    r3 = zeros(NP, 1);
    
    for i = 1:NP
        % Select r1 (different from i)
        rnd = rand();
        r1(i) = find(cum_probs >= rnd, 1);
        while r1(i) == i
            rnd = rand();
            r1(i) = find(cum_probs >= rnd, 1);
        end
        
        % Select r2 (different from i and r1)
        rnd = rand();
        r2(i) = find(cum_probs >= rnd, 1);
        while r2(i) == i || r2(i) == r1(i)
            rnd = rand();
            r2(i) = find(cum_probs >= rnd, 1);
        end
        
        % Select r3 (different from i, r1, and r2)
        rnd = rand();
        r3(i) = find(cum_probs >= rnd, 1);
        while r3(i) == i || r3(i) == r1(i) || r3(i) == r2(i)
            rnd = rand();
            r3(i) = find(cum_probs >= rnd, 1);
        end
    end
    
    % Compute adaptive F and lambda
    F = 0.5 + 0.3 * (1 - weights);
    lambda = 0.5 * (1 - norm_fits);
    
    % Mutation
    x_r1 = popdecs(r1, :);
    x_r2 = popdecs(r2, :);
    x_r3 = popdecs(r3, :);
    
    diff = x_r2 - x_r3;
    best_diff = x_best - popdecs;
    
    % Vectorized mutation with adaptive parameters
    v = x_r1 + F .* diff + lambda .* best_diff;
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
end
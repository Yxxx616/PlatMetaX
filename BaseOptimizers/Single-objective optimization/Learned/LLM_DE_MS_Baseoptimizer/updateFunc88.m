function [offspring] = updateFunc88(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    if c_max == 0
        c_max = 1;
    end
    
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / c_max;
    
    % Calculate selection weights
    weights = (1 - norm_fits) .* (1 + norm_cons);
    weights = weights / sum(weights);  % Normalize to probabilities
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Parameters
    CR = 0.9;
    
    % Pre-compute cumulative probabilities for selection
    cum_probs = cumsum(weights);
    
    for i = 1:NP
        % Select three distinct vectors using weighted probabilities
        r = zeros(1,3);
        for k = 1:3
            rnd = rand();
            r(k) = find(cum_probs >= rnd, 1);
            while any(r(1:k-1) == r(k)) || r(k) == i
                rnd = rand();
                r(k) = find(cum_probs >= rnd, 1);
            end
        end
        
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        x_r3 = popdecs(r(3), :);
        
        % Adaptive scaling factors
        F = 0.5 + 0.3 * norm_cons(i);
        lambda = 0.5 * norm_fits(i);
        
        % Mutation
        v = x_r1 + F * (x_r2 - x_r3) + lambda * (x_best - popdecs(i, :));
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        % Create offspring
        offspring(i, :) = popdecs(i, :);
        offspring(i, mask) = v(mask);
    end
end
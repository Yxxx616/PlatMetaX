function [offspring] = updateFunc87(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    if c_max == 0
        c_max = 1;  % Avoid division by zero
    end
    
    % Sort population by fitness
    [~, idx] = sort(popfits);
    sorted_pop = popdecs(idx, :);
    sorted_fits = popfits(idx);
    sorted_cons = cons(idx);
    
    % Determine segments
    n_best = max(1, round(0.1*NP));
    n_mid = max(1, round(0.6*NP));
    best_range = 1:n_best;
    mid_range = (n_best+1):(n_best+n_mid);
    
    CR = 0.9;  % Crossover probability
    
    for i = 1:NP
        % Select vectors based on fitness ranking
        best_idx = best_range(randi(length(best_range)));
        mid_idx1 = mid_range(randi(length(mid_range)));
        mid_idx2 = mid_range(randi(length(mid_range)));
        while mid_idx1 == mid_idx2
            mid_idx2 = mid_range(randi(length(mid_range)));
        end
        
        x_best = sorted_pop(best_idx, :);
        x_mid1 = sorted_pop(mid_idx1, :);
        x_mid2 = sorted_pop(mid_idx2, :);
        
        % Select two random distinct vectors
        r = randperm(NP, 2);
        while any(r == i)
            r = randperm(NP, 2);
        end
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        
        % Compute adaptive scaling factors
        F1 = 0.5 * (1 - (sorted_fits(i) - f_min)/(f_max - f_min));
        F2 = 0.8 * (abs(sorted_cons(i))/c_max);
        
        % Generate mutant vector with constraint-based perturbation
        v = x_best + F1*(x_mid1 - x_mid2) + F2*(x_r1 - x_r2)*(1 + sorted_cons(i)/c_max);
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        % Create offspring
        offspring(i, :) = sorted_pop(i, :);
        offspring(i, mask) = v(mask);
    end
end
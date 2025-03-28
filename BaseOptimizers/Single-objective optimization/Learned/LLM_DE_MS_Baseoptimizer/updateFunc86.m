function [offspring] = updateFunc86(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness
    [~, idx] = sort(popfits);
    sorted_pop = popdecs(idx, :);
    sorted_fits = popfits(idx);
    sorted_cons = cons(idx);
    
    % Determine segments
    n_best = max(1, round(0.1*NP));
    n_mid = max(1, round(0.6*NP));
    
    % Precompute min/max for normalization
    f_min = min(popfits);
    f_max = max(popfits);
    cons_max = max(abs(cons));
    
    CR = 0.9;  % Crossover probability
    
    for i = 1:NP
        % Select vectors based on fitness ranking
        best_idx = randi(n_best);
        mid_idx = randi([n_best+1, n_best+n_mid]);
        worst_idx = randi([NP-n_best+1, NP]);
        
        x_best = sorted_pop(best_idx, :);
        x_mid = sorted_pop(mid_idx, :);
        x_worst = sorted_pop(worst_idx, :);
        
        % Select two random distinct vectors
        r = randperm(NP, 2);
        while any(r == i)
            r = randperm(NP, 2);
        end
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        
        % Compute adaptive scaling factors
        F1 = 0.5 * (1 + sorted_cons(i)/cons_max);
        F2 = 0.5 * (1 - (sorted_fits(i) - f_min)/(f_max - f_min));
        
        % Generate mutant vector
        v = x_best + F1*(x_mid - x_worst) + F2*(x_r1 - x_r2);
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        % Create offspring
        offspring(i, :) = sorted_pop(i, :);
        offspring(i, mask) = v(mask);
    end
end
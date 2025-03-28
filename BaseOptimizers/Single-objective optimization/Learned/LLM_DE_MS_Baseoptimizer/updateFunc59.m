function [offspring] = updateFunc59(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best and most feasible solutions
    [~, bestIdx] = min(popfits);
    [~, feasIdx] = min(abs(cons));
    x_best = popdecs(bestIdx, :);
    x_feas = popdecs(feasIdx, :);
    f_best = popfits(bestIdx);
    c_feas = cons(feasIdx);
    
    c_max = max(abs(cons));  % Maximum constraint violation
    
    for i = 1:NP
        % Select three distinct random vectors
        idxs = randperm(NP, 3);
        while any(idxs == i)
            idxs = randperm(NP, 3);
        end
        x_r1 = popdecs(idxs(1), :);
        x_r2 = popdecs(idxs(2), :);
        x_r3 = popdecs(idxs(3), :);
        
        % Current individual
        x_i = popdecs(i, :);
        f_i = popfits(i);
        c_i = cons(i);
        
        % Calculate adaptive parameters
        if c_max > 0
            ratio = abs(c_i)/c_max;
            theta = pi * ratio;
            alpha = 0.5 * (1 + cos(theta));
            beta = 0.5 * (1 - cos(theta));
            F = 0.5 * (1 + tanh(1 - ratio));
        else
            alpha = 1;
            beta = 0;
            F = 0.5;
        end
        
        % Calculate direction vectors
        d_fit = (x_best - x_i) .* (f_best - f_i);
        d_cons = (x_feas - x_i) .* (c_i - c_feas);
        
        % Generate mutant vector
        offspring(i,:) = x_r1 + F*(x_r2 - x_r3) + alpha*d_fit + beta*d_cons;
    end
end
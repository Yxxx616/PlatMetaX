function [offspring] = updateFunc56(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best and most feasible solutions
    [~, bestIdx] = min(popfits);
    [~, feasIdx] = min(abs(cons));
    x_best = popdecs(bestIdx, :);
    x_feas = popdecs(feasIdx, :);
    f_best = popfits(bestIdx);
    c_feas = cons(feasIdx);
    
    F = 0.5;  % Mutation factor
    alpha = 0.3;  % Fitness direction weight
    beta = 0.2;   % Constraint direction weight
    
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
        
        % Calculate direction vectors
        d_fit = (x_best - x_i) * (f_best - f_i);
        d_cons = (x_feas - x_i) * (c_i - c_feas);
        
        % Generate mutant vector
        offspring(i,:) = x_r1 + F*(x_r2 - x_r3) + alpha*d_fit + beta*d_cons;
    end
end
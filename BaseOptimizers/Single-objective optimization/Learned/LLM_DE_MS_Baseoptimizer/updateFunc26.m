function [offspring] = updateFunc26(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    CR = 0.9;
    
    % Find best individual based on fitness
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Calculate fitness ranks (lower is better)
    [~, ~, fit_ranks] = unique(popfits);
    fit_probs = 1./(fit_ranks + 1e-10);
    fit_probs = fit_probs/sum(fit_probs);
    
    % Calculate constraint violation ranks (lower is better)
    abs_cons = abs(cons);
    [~, ~, cons_ranks] = unique(abs_cons);
    cons_probs = 1./(cons_ranks + 1e-10);
    cons_probs = cons_probs/sum(cons_probs);
    
    % Generate offspring
    for i = 1:NP
        % Select indices
        r1 = best_idx;
        
        % Select r2 based on fitness
        r2 = randsample(NP, 1, true, fit_probs);
        while r2 == i
            r2 = randsample(NP, 1, true, fit_probs);
        end
        
        % Select r3 based on constraint violation
        r3 = randsample(NP, 1, true, cons_probs);
        while r3 == i || r3 == r2
            r3 = randsample(NP, 1, true, cons_probs);
        end
        
        % Adaptive lambda based on constraint violation
        max_cons = max(abs(cons));
        lambda = 0.5 * (1 + cons(i)/max_cons);
        
        % Mutation
        v = popdecs(r1,:) + F*(popdecs(r2,:) - popdecs(r3,:)) + ...
            lambda*(x_best - popdecs(i,:));
        
        % Crossover
        j_rand = randi(D);
        for j = 1:D
            if rand() < CR || j == j_rand
                offspring(i,j) = v(j);
            else
                offspring(i,j) = popdecs(i,j);
            end
        end
    end
end
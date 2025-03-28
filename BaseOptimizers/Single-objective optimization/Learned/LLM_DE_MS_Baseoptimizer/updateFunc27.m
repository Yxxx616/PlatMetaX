% MATLAB Code
function [offspring] = updateFunc27(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    CR = 0.9;
    alpha = 0.7;
    beta = 0.3;
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Calculate fitness probabilities
    [~, ~, fit_ranks] = unique(popfits);
    fit_probs = 1./(fit_ranks + 1e-10);
    fit_probs = fit_probs/sum(fit_probs);
    
    % Calculate constraint probabilities
    abs_cons = abs(cons);
    [~, ~, cons_ranks] = unique(abs_cons);
    cons_probs = 1./(cons_ranks + 1e-10);
    cons_probs = cons_probs/sum(cons_probs);
    
    for i = 1:NP
        % Select x_fit based on fitness
        x_fit_idx = randsample(NP, 1, true, fit_probs);
        while x_fit_idx == i
            x_fit_idx = randsample(NP, 1, true, fit_probs);
        end
        
        % Select x_cons based on constraints
        x_cons_idx = randsample(NP, 1, true, cons_probs);
        while x_cons_idx == i || x_cons_idx == x_fit_idx
            x_cons_idx = randsample(NP, 1, true, cons_probs);
        end
        
        % Select random individual
        x_rand_idx = randi(NP);
        while x_rand_idx == i || x_rand_idx == x_fit_idx || x_rand_idx == x_cons_idx
            x_rand_idx = randi(NP);
        end
        
        % Mutation
        v = x_best + F * (popdecs(x_fit_idx,:) - popdecs(x_rand_idx,:)) + ...
            beta * (x_best - popdecs(x_cons_idx,:));
        
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
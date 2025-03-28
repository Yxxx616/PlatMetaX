% MATLAB Code
function [offspring] = updateFunc28(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Adaptive parameters
    F1 = 0.5 + 0.3 * rand();
    F2 = 0.5 + 0.3 * rand();
    F3 = 0.3 + 0.2 * rand();
    CR = 0.85;
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Calculate fitness probabilities (inverse ranking)
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    fit_probs = 1./(ranks + 1e-10);
    fit_probs = fit_probs/sum(fit_probs);
    
    % Calculate constraint probabilities (inverse ranking)
    abs_cons = abs(cons);
    [~, sorted_idx] = sort(abs_cons);
    cons_ranks = zeros(NP,1);
    cons_ranks(sorted_idx) = 1:NP;
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
        
        % Select two distinct random individuals
        candidates = setdiff(1:NP, [i, x_fit_idx, x_cons_idx]);
        x_rand_idx = candidates(randi(length(candidates)));
        x_rand2_idx = candidates(randi(length(candidates)));
        while x_rand2_idx == x_rand_idx
            x_rand2_idx = candidates(randi(length(candidates)));
        end
        
        % Mutation
        v = popdecs(i,:) + F1 * (x_best - popdecs(i,:)) + ...
            F2 * (popdecs(x_fit_idx,:) - popdecs(x_rand_idx,:)) + ...
            F3 * (popdecs(x_cons_idx,:) - popdecs(x_rand2_idx,:));
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = v(mask);
    end
end
% MATLAB Code
function [offspring] = updateFunc30(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Adaptive parameters with dynamic ranges
    F1 = 0.7 + 0.1 * rand();
    F2 = 0.6 + 0.2 * rand();
    F3 = 0.5 + 0.2 * rand();
    F4 = 0.3 + 0.2 * rand();
    CR = 0.85 + 0.1 * rand();
    
    % Find best individual based on fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
    else
        [~, best_idx] = min(cons);
    end
    x_best = popdecs(best_idx, :);
    
    % Normalized fitness weights
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + 1e-10);
    fit_weights = 1 - f_norm; % Higher weight for better fitness
    
    % Normalized constraint weights
    c_norm = (abs(cons) - min(abs(cons))) / (max(abs(cons)) - min(abs(cons)) + 1e-10);
    cons_weights = 1 - c_norm; % Higher weight for smaller constraint violation
    
    % Combined selection probabilities
    alpha = 0.6; % Fitness importance factor
    selection_probs = alpha * fit_weights + (1-alpha) * cons_weights;
    selection_probs = selection_probs / sum(selection_probs);
    
    for i = 1:NP
        % Select three distinct random individuals using weighted probability
        candidates = setdiff(1:NP, i);
        selected = randsample(candidates, 3, true, selection_probs(candidates));
        x_r1 = popdecs(selected(1), :);
        x_r2 = popdecs(selected(2), :);
        x_r3 = popdecs(selected(3), :);
        
        % Select one more individual randomly for exploration
        remaining = setdiff(candidates, selected);
        x_r4 = popdecs(remaining(randi(length(remaining))), :);
        
        % Calculate adaptive weights for current individual
        w_f = 1 / (1 + exp(-(popfits(i) - popfits(selected(1))));
        w_c = 1 / (1 + exp(-(abs(cons(i)) - abs(cons(selected(2))))));
        
        % Enhanced mutation strategy
        v = popdecs(i,:) + F1 * (x_best - popdecs(i,:)) + ...
            F2 * w_f * (x_r1 - x_r2) + ...
            F3 * w_c * (x_r2 - x_r3) + ...
            F4 * (x_r3 - x_r4);
        
        % Dynamic crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = v(mask);
    end
end
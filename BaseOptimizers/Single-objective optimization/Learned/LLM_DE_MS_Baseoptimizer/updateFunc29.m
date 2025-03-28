% MATLAB Code
function [offspring] = updateFunc29(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Adaptive parameters
    F1 = 0.6 + 0.2 * rand();
    F2 = 0.5 + 0.3 * rand();
    F3 = 0.4 + 0.3 * rand();
    F4 = 0.1 + 0.1 * rand();
    CR = 0.9;
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Calculate fitness weights
    f_min = min(popfits);
    f_max = max(popfits);
    sigma_f = (f_max - f_min) / 10;
    fit_weights = 1 ./ (1 + exp(-(popfits - mean(popfits)) / (sigma_f + 1e-10)));
    
    % Calculate constraint weights
    abs_cons = abs(cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons);
    sigma_c = (c_max - c_min) / 10;
    cons_weights = 1 ./ (1 + exp(-(abs_cons - mean(abs_cons)) / (sigma_c + 1e-10)));
    
    for i = 1:NP
        % Select individuals based on fitness weights
        fit_probs = fit_weights / sum(fit_weights);
        x_fit_idx = randsample(NP, 1, true, fit_probs);
        
        % Select individuals based on constraint weights
        cons_probs = cons_weights / sum(cons_weights);
        x_cons_idx = randsample(NP, 1, true, cons_probs);
        
        % Select four distinct random individuals
        candidates = setdiff(1:NP, [i, x_fit_idx, x_cons_idx]);
        rand_idx = candidates(randperm(length(candidates), 4));
        x_rand1 = popdecs(rand_idx(1), :);
        x_rand2 = popdecs(rand_idx(2), :);
        x_rand3 = popdecs(rand_idx(3), :);
        x_rand4 = popdecs(rand_idx(4), :);
        
        % Calculate weights for current individual
        w_fit = 1 / (1 + exp(-(popfits(i) - popfits(x_fit_idx))/(sigma_f + 1e-10)));
        w_cons = 1 / (1 + exp(-(abs_cons(i) - abs_cons(x_cons_idx))/(sigma_c + 1e-10)));
        
        % Mutation
        v = popdecs(i,:) + F1 * (x_best - popdecs(i,:)) + ...
            F2 * w_fit * (popdecs(x_fit_idx,:) - x_rand1) + ...
            F3 * w_cons * (popdecs(x_cons_idx,:) - x_rand2) + ...
            F4 * (x_rand3 - x_rand4);
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = v(mask);
    end
end
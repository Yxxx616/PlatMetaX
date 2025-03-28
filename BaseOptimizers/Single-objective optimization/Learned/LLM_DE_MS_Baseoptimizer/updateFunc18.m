% MATLAB Code
function [offspring] = updateFunc18(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations
    max_cons = max(abs(cons));
    cons_norm = abs(cons) / (max_cons + 1e-12);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    ranks = 1:NP;
    
    % Determine segments
    best_idx = sorted_idx(1:max(1,round(0.2*NP)));
    mid_idx = sorted_idx(round(0.2*NP)+1:round(0.8*NP));
    worst_idx = sorted_idx(round(0.8*NP)+1:end);
    
    for i = 1:NP
        % Select base vectors from different segments
        x_best = popdecs(best_idx(randi(length(best_idx))), :);
        x_mid = popdecs(mid_idx(randi(length(mid_idx))), :);
        x_worst = popdecs(worst_idx(randi(length(worst_idx))), :);
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        r = candidates(randperm(length(candidates), 2));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        
        % Adaptive scaling factors
        F1 = 0.5 + 0.3 * (1 - ranks(i)/NP);
        F2 = 0.3 * (1 + cons_norm(i));
        
        % Mutation
        offspring(i,:) = x_best + F1.*(x_mid - x_worst) + F2.*(x_r1 - x_r2);
    end
    
    % Bound handling
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end
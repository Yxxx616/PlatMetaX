% MATLAB Code
function [offspring] = updateFunc1436(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    feasible = cv <= 0;
    cv_norm = cv / (max(cv) + eps);
    
    % Normalize fitness (minimization)
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Select elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute feasible center (weighted average)
    if any(feasible)
        weights = 1./(1 + abs(popfits(feasible)) + cv(feasible));
        weights = weights / sum(weights);
        x_feas = sum(popdecs(feasible,:) .* weights(:), 1);
    else
        weights = 1./(1 + abs(popfits) + cv);
        weights = weights / sum(weights);
        x_feas = sum(popdecs .* weights(:), 1);
    end
    
    % Generate random indices for diversity component
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    invalid = (rand_idx1 == rand_idx2) | (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)');
    while any(invalid)
        rand_idx1(invalid) = randi(NP, sum(invalid), 1);
        rand_idx2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (rand_idx1 == rand_idx2) | (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)');
    end
    
    % Adaptive weights
    w1 = 1./(1 + f_norm + cv_norm);
    w2 = 1./(1 + abs(popfits) + cv);
    w3 = 1./(1 + 0.5*f_norm + 0.5*cv_norm);
    w_total = w1 + w2 + w3;
    w1 = w1 ./ w_total;
    w2 = w2 ./ w_total;
    w3 = w3 ./ w_total;
    
    % Dynamic scaling factors
    F = 0.5 * (1 - cv_norm) .* (0.5 + 0.5*rand(NP,1));
    
    % Mutation components
    diff_elite = x_elite - popdecs;
    diff_feas = x_feas - popdecs;
    diff_div = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    % Combined mutation
    mutants = popdecs + F.*(w1.*diff_elite + w2.*diff_feas + w3.*diff_div);
    
    % Boundary handling with reflection
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = 2*lb(j) - mutants(below,j);
        mutants(above,j) = 2*ub(j) - mutants(above,j);
        
        % Reinitialize if still out of bounds
        still_bad = (mutants(:,j) < lb(j)) | (mutants(:,j) > ub(j));
        mutants(still_bad,j) = lb(j) + (ub(j)-lb(j)).*rand(sum(still_bad),1);
    end
    
    % Dynamic crossover rate
    CR = 0.9 * (1 - cv_norm);
    CR = min(max(CR, 0.1), 0.9);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Special handling for worst solutions
    bad_sols = (f_norm + cv_norm) > 1.5;
    if any(bad_sols)
        offspring(bad_sols,:) = x_feas + (ub - lb).*(rand(sum(bad_sols),D)-0.5)*0.2;
    end
end
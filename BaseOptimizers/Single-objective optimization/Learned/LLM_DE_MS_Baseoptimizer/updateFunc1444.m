% MATLAB Code
function [offspring] = updateFunc1444(popdecs, popfits, cons)
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
    
    % Select elite individual considering both fitness and constraints
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute feasible center (weighted average)
    if any(feasible)
        weights = exp(-abs(popfits(feasible))) .* exp(-cv(feasible));
    else
        weights = exp(-abs(popfits)) .* exp(-cv);
    end
    weights = weights / (sum(weights) + eps);
    x_feas = sum(popdecs .* weights(:), 1);
    
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
    w_elite = 1./(1 + f_norm + cv_norm);
    w_feas = 1./(1 + cv_norm);
    w_div = 1./(1 + f_norm);
    w_total = w_elite + w_feas + w_div + eps;
    w_elite = w_elite ./ w_total;
    w_feas = w_feas ./ w_total;
    w_div = w_div ./ w_total;
    
    % Dynamic scaling factor (cosine decreasing)
    iter = randi(100); % Current iteration (simulated)
    maxiter = 100;     % Max iterations (simulated)
    F = 0.5 * (1 + cos(pi * iter/maxiter));
    
    % Mutation components
    diff_elite = x_elite - popdecs;
    diff_feas = x_feas - popdecs;
    diff_div = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    % Combined mutation with adaptive weights
    mutants = w_elite.*(popdecs + F.*diff_elite) + ...
              w_feas.*(popdecs + F.*diff_feas) + ...
              w_div.*(popdecs + F.*diff_div.*(1 - cv_norm));
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    
    % Dynamic crossover rate
    CR = 0.9 * (1 - sqrt(cv_norm));
    CR = min(max(CR, 0.1), 0.9);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Additional perturbation for highly constrained solutions
    highly_constrained = cv_norm > 0.8;
    if any(highly_constrained)
        sigma = 0.2 * (ub - lb);
        perturbation = randn(sum(highly_constrained),D) .* sigma;
        offspring(highly_constrained,:) = x_feas + perturbation;
        offspring(highly_constrained,:) = min(max(offspring(highly_constrained,:), lb), ub);
    end
end
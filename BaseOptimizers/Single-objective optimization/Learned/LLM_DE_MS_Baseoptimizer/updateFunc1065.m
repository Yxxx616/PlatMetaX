% MATLAB Code
function [offspring] = updateFunc1065(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weighted center considering both fitness and constraints
    c_max = max(abs(cons)) + eps;
    f_min = min(popfits);
    f_max = max(popfits);
    
    % Normalized fitness and constraints
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / c_max;
    
    % Combined weights
    weights = (1 - norm_fits) + norm_cons;
    weights = weights / sum(weights);
    x_weighted = weights' * popdecs;
    
    % 2. Adaptive scaling factors based on constraints
    F = 0.5 * (1 + tanh(norm_cons));
    F = F(:, ones(1, D));
    
    % 3. Select random vectors (different indices)
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    
    % 4. Create mutant vectors
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutants = x_weighted(ones(NP,1), :) + ...
              F .* diff1 + ...
              (1-F) .* diff2;
    
    % 5. Constraint-aware crossover
    CR = 0.9 - 0.4 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % 8. Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end
% MATLAB Code
function [offspring] = updateFunc1064(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Find best solution (considering both fitness and constraints)
    [~, best_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    
    % 2. Normalize constraints and fitness
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    % 3. Compute adaptive weights and scaling factors
    weights = 0.5 * (1 - norm_fits);
    F = 0.5 * (1 - tanh(norm_cons));
    
    % 4. Select random indices (different from current and best)
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    
    % 5. Create mutant vectors
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutants = x_best(ones(NP,1), :) + ...
              F(:, ones(1,D)) .* diff1 + ...
              (1-F(:, ones(1,D))) .* weights(:, ones(1,D)) .* diff2;
    
    % 6. Dynamic crossover with constraint adaptation
    CR = 0.9 - 0.5 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % 9. Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end
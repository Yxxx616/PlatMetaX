% MATLAB Code
function [offspring] = updateFunc1062(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Find best solution (considering both fitness and constraints)
    [~, best_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    
    % 2. Normalize constraints
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    % 3. Select random indices (different from current and best)
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    
    % 4. Compute dynamic scaling factors
    F1 = 0.5 + 0.3 * tanh(1 - norm_cons);
    F2 = 0.3 + 0.2 * tanh(norm_cons);
    
    % 5. Create mutant vectors
    mutants = x_best + ...
              F1(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:)) + ...
              F2(:, ones(1,D)) .* (popdecs(r3,:) - popdecs(r4,:));
    
    % 6. Add fitness-weighted guidance
    max_fit = max(abs(popfits)) + eps;
    weights = 0.1 * (1 - abs(popfits)/max_fit);
    mutants = mutants + weights(:, ones(1,D)) .* (x_best - popdecs);
    
    % 7. Dynamic crossover
    CR = 0.9 - 0.4 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % 10. Ensure final bounds
    offspring = min(max(offspring, lb_rep), ub_rep);
end
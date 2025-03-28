% MATLAB Code
function [offspring] = updateFunc1607(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 1e6*abs(cons));
        elite = popdecs(elite_idx,:);
    end
    
    % 2. Compute adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.4 + 0.3 * (popfits - f_min) / (f_max - f_min + eps);
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    G = 0.2 * exp(-c_abs / (c_max + eps));
    
    % 3. Direction-guided mutation with 4 random vectors
    idx = randperm(NP, 4*NP);
    r1 = reshape(idx(1:NP), [], 1);
    r2 = reshape(idx(NP+1:2*NP), [], 1);
    r3 = reshape(idx(2*NP+1:3*NP), [], 1);
    r4 = reshape(idx(3*NP+1:4*NP), [], 1);
    
    mutation = elite(ones(NP,1), :) + ...
               F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               G.*(popdecs(r3,:) - popdecs(r4,:));
    
    % 4. Constraint-aware crossover rate
    c_min = min(cons);
    c_max = max(cons);
    CR = 0.9 - 0.4 * (cons - c_min) / (c_max - c_min + eps);
    
    % 5. Crossover with opposition-based components
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Opposition-based components (10% probability)
    oppo_pop = lb + ub - popdecs;
    oppo_mask = rand(NP,D) < 0.1;
    
    offspring = popdecs .* (~mask) + mutation .* mask;
    offspring = offspring .* (~oppo_mask) + oppo_pop .* oppo_mask;
    
    % 6. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + rand(sum(below(:)),1) .* (ub_rep(below) - lb_rep(below));
    offspring(above) = lb_rep(above) - rand(sum(above(:)),1) .* (ub_rep(above) - lb_rep(above));
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
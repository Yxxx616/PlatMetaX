% MATLAB Code
function [offspring] = updateFunc1608(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 1e6*abs(cons));
        elite = popdecs(elite_idx,:);
    end
    
    % 2. Adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.5 * (1 + (popfits - f_min) ./ (f_max - f_min + eps));
    
    c_abs = abs(cons);
    c_max = max(c_abs);
    G = 0.5 * exp(-c_abs ./ (c_max + eps));
    
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
    CR = 0.9 - 0.5 * (c_abs ./ (c_max + eps));
    
    % 5. Binomial crossover
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % 6. Dynamic boundary handling with reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc1329(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select reference points
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(find(feasible, 1, 'first'), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % 2. Calculate direction vectors
    directions = zeros(NP, D);
    for i = 1:NP
        if feasible(i)
            directions(i,:) = x_best - popdecs(i,:);
        else
            directions(i,:) = (x_best + 0.1*randn(1,D)) - popdecs(i,:);
        end
    end
    
    % 3. Compute adaptive scaling factors
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    max_cons = max(cons) + 1e-6;
    F = 0.5 + 0.4 * (1 - rank/NP) .* (1 - cons./max_cons);
    
    % 4. Generate mutant vectors with differential component
    idx = randperm(NP);
    a = idx(1:NP);
    b = idx(NP+1:2*NP);
    mutants = popdecs + F.*directions + 0.2*randn(NP,1).*(popdecs(a,:) - popdecs(b,:));
    
    % 5. Adaptive crossover
    CR = 0.9 - 0.5*(rank/NP);
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Constraint repair
    violate = cons > 0;
    for i = 1:NP
        if violate(i)
            if rand() < 0.7
                offspring(i,:) = x_best + 0.5*(x_best - offspring(i,:));
            else
                offspring(i,:) = lb + (ub - lb).*rand(1,D);
            end
        end
    end
    
    % 7. Boundary handling with reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    offspring = min(max(offspring, lb), ub);
end
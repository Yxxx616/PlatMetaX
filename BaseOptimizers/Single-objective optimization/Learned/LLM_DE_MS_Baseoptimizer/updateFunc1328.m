% MATLAB Code
function [offspring] = updateFunc1328(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify reference points
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(find(feasible, best_idx, 'first'), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    [~, least_violated_idx] = min(cons);
    x_least = popdecs(least_violated_idx, :);
    
    % 2. Calculate direction vectors
    directions = zeros(NP, D);
    for i = 1:NP
        if cons(i) <= 0
            directions(i,:) = x_best - popdecs(i,:);
        else
            directions(i,:) = x_least - popdecs(i,:);
        end
    end
    
    % 3. Compute adaptive scaling factors
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    max_cons = max(cons);
    F = 0.5 + 0.3 * (1 - rank/NP) .* (1 - cons./(max_cons + 1));
    
    % 4. Generate mutant vectors
    alpha = 0.2 * rand(NP, 1);
    mutants = popdecs + F.*directions + alpha.*(lb + ub - 2*popdecs);
    
    % 5. Add elite perturbation
    [~, sorted_idx] = sort(popfits);
    elite_size = max(2, ceil(0.2*NP));
    elite = popdecs(sorted_idx(1:elite_size), :);
    e_idx = randi(elite_size, NP, 1);
    mutants = mutants + 0.1*(elite(e_idx,:) - popdecs);
    
    % 6. Adaptive crossover
    CR = 0.9 - 0.4*(rank/NP);
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Constraint repair with reflection
    violate = cons > 0;
    for i = 1:NP
        if violate(i)
            if rand() < 0.6
                offspring(i,:) = x_best + 0.5*(x_best - offspring(i,:));
            else
                offspring(i,:) = lb + (ub - lb).*rand(1,D);
            end
        end
    end
    
    % 8. Boundary handling
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    offspring = min(max(offspring, lb), ub);
end
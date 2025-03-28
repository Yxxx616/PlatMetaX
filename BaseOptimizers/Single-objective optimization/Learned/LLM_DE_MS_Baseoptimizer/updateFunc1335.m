% MATLAB Code
function [offspring] = updateFunc1335(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    epsilon = 1e-12;
    
    % 1. Sort solutions based on fitness and constraints
    [~, sorted_idx] = sortrows([popfits, cons], [1 2]);
    popdecs_sorted = popdecs(sorted_idx, :);
    popfits_sorted = popfits(sorted_idx);
    cons_sorted = cons(sorted_idx);
    
    % 2. Select elite solutions (top 30%)
    elite_num = max(3, ceil(0.3 * NP));
    elites = popdecs_sorted(1:elite_num, :);
    elite_fits = popfits_sorted(1:elite_num);
    
    % 3. Calculate adaptive parameters
    cons_max = max(abs(cons));
    f_max = max(popfits);
    f_min = min(popfits);
    
    % Adaptive scaling factors
    F_base = 0.5;
    F_adapt = 0.3 * (popfits - f_min) ./ (f_max - f_min + epsilon);
    F = min(max(F_base + F_adapt, 0.4), 0.9);
    
    % Adaptive crossover rates
    CR = 0.9 - 0.5 * (abs(cons) ./ (cons_max + epsilon));
    
    % 4. Generate offspring
    for i = 1:NP
        % Select distinct random indices
        candidates = setdiff(1:NP, i);
        idxs = candidates(randperm(length(candidates), 4));
        
        % Select random elite based on fitness
        elite_probs = exp(-elite_fits) / sum(exp(-elite_fits));
        elite_idx = find(rand() <= cumsum(elite_probs), 1);
        x_elite = elites(elite_idx, :);
        
        % Best and worst solutions
        x_best = popdecs_sorted(1, :);
        x_worst = popdecs_sorted(end, :);
        
        % Enhanced mutation
        mutant = x_elite + F(i) .* (popdecs(idxs(1), :) - popdecs(idxs(2), :)) + ...
                0.5 * (x_best - x_worst);
        
        % Adaptive crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR(i);
        mask(j_rand) = true;
        
        % Create trial vector
        trial = popdecs(i, :);
        trial(mask) = mutant(mask);
        
        % Constraint repair mechanism
        if cons(i) > 0
            beta = 0.8 * min(1, abs(cons(i))/(cons_max + epsilon);
            trial = beta * x_elite + (1-beta) * trial;
        end
        
        offspring(i, :) = trial;
    end
    
    % Boundary handling with random reinitialization
    below = offspring < lb;
    above = offspring > ub;
    for j = 1:D
        if any(below(:,j))
            offspring(below(:,j),j) = lb(j) + rand(sum(below(:,j)),1) .* (ub(j)-lb(j));
        end
        if any(above(:,j))
            offspring(above(:,j),j) = lb(j) + rand(sum(above(:,j)),1) .* (ub(j)-lb(j));
        end
    end
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end
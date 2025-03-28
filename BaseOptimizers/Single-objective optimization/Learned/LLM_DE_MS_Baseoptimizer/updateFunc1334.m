% MATLAB Code
function [offspring] = updateFunc1334(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    epsilon = 1e-6;
    
    % 1. Sort solutions based on fitness and constraints
    [~, sorted_idx] = sortrows([popfits, cons], [1 2]);
    popdecs_sorted = popdecs(sorted_idx, :);
    popfits_sorted = popfits(sorted_idx);
    cons_sorted = cons(sorted_idx);
    
    % 2. Select elite solutions (top 20%)
    elite_num = max(3, ceil(0.2 * NP));
    elites = popdecs_sorted(1:elite_num, :);
    elite_fits = popfits_sorted(1:elite_num);
    
    % 3. Calculate adaptive parameters
    cons_max = max(cons);
    cons_min = min(cons);
    f_max = max(popfits);
    f_min = min(popfits);
    
    % Constraint-aware scaling factors
    F_base = 0.5;
    F_cons = 0.3 * (cons_max - cons) ./ (cons_max - cons_min + epsilon);
    F = min(max(F_base + F_cons, 0.4), 0.8);
    
    % Fitness-based weights
    alpha = 0.7;
    fit_weights = (popfits - f_min) ./ (f_max - f_min + epsilon);
    
    % 4. Generate offspring
    for i = 1:NP
        % Select distinct random indices (including elites)
        candidates = setdiff(1:NP, i);
        idxs = candidates(randperm(length(candidates), 4));
        
        % Select random elite based on fitness
        elite_probs = exp(-elite_fits) / sum(exp(-elite_fits));
        elite_idx = find(rand() <= cumsum(elite_probs), 1);
        x_elite = elites(elite_idx, :);
        
        % Best and worst solutions
        x_best = popdecs_sorted(1, :);
        x_worst = popdecs_sorted(end, :);
        
        % Enhanced mutation with fitness-weighted direction
        mutant = x_elite + F(i) .* (popdecs(idxs(1), :) - popdecs(idxs(2), :)) + ...
                alpha .* (x_best - x_worst) .* (1 - fit_weights(i));
        
        % Adaptive crossover with constraint awareness
        CR_i = 0.9 - 0.4*(cons(i)/(cons_max + epsilon)) + 0.1*rand();
        j_rand = randi(D);
        mask = rand(1, D) < CR_i;
        mask(j_rand) = true;
        
        % Create trial vector
        trial = popdecs(i, :);
        trial(mask) = mutant(mask);
        
        % Constraint repair mechanism
        if cons(i) > 0
            beta = 0.7 * min(1, cons(i)/cons_max);
            trial = beta * x_elite + (1-beta) * trial;
        end
        
        offspring(i, :) = trial;
    end
    
    % Boundary handling with reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    
    % Final clamping to ensure feasibility
    offspring = min(max(offspring, lb), ub);
end
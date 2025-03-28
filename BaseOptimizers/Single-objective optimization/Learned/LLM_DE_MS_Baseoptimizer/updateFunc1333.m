% MATLAB Code
function [offspring] = updateFunc1333(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Sort solutions based on fitness and constraints
    [~, sorted_idx] = sortrows([popfits, cons], [1 2]);
    popdecs_sorted = popdecs(sorted_idx, :);
    cons_sorted = cons(sorted_idx);
    
    % 2. Select elite solutions (top 30%)
    elite_num = max(3, ceil(0.3 * NP));
    elites = popdecs_sorted(1:elite_num, :);
    
    % 3. Calculate adaptive parameters
    cons_max = max(cons);
    cons_min = min(cons);
    f_max = max(popfits);
    f_min = min(popfits);
    epsilon = 1e-6;
    
    % Scaling factors with constraint awareness
    F_base = 0.5 * ones(NP, 1);
    F_cons = 0.5 * (cons_max - cons) ./ (cons_max - cons_min + epsilon);
    F = min(max(F_base + F_cons, 0.3), 0.9);
    
    % Fitness-based weights
    alpha = 0.7;
    fit_weights = (popfits - f_min) ./ (f_max - f_min + epsilon);
    
    % 4. Generate offspring
    for i = 1:NP
        % Select distinct random indices
        idxs = randperm(NP, 4);
        while any(idxs == i)
            idxs = randperm(NP, 4);
        end
        
        % Select random elite
        elite_idx = randi(size(elites, 1));
        x_elite = elites(elite_idx, :);
        
        % Best and worst solutions
        x_best = popdecs_sorted(1, :);
        x_worst = popdecs_sorted(end, :);
        
        % Enhanced mutation with fitness-weighted direction
        mutant = x_elite + F(i) .* (popdecs(idxs(1), :) - popdecs(idxs(2), :)) + ...
                alpha .* (x_best - x_worst) .* (1 - fit_weights(i));
        
        % Adaptive crossover
        CR_i = 0.9 - 0.4*(cons(i)/(cons_max + epsilon)) + 0.1*rand();
        j_rand = randi(D);
        mask = rand(1, D) < CR_i;
        mask(j_rand) = true;
        
        % Create trial vector
        trial = popdecs(i, :);
        trial(mask) = mutant(mask);
        
        % Enhanced constraint repair mechanism
        if cons(i) > 0
            beta = 0.7 * min(1, cons(i)/cons_max);
            trial = beta * x_elite + (1-beta) * trial;
        end
        
        offspring(i, :) = trial;
    end
    
    % Boundary handling with reflection and clamping
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = lb(below) + 0.5 * rand(sum(below(:)),1) .* (ub(below) - lb(below));
    offspring(above) = ub(above) - 0.5 * rand(sum(above(:)),1) .* (ub(above) - lb(above));
    offspring = min(max(offspring, lb), ub);
end
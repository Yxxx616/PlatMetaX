% MATLAB Code
function [offspring] = updateFunc1331(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Elite selection (top 30% solutions)
    [~, sorted_idx] = sort([popfits, cons], [1 2]);
    elite_num = ceil(0.3 * NP);
    elites = popdecs(sorted_idx(1:elite_num), :);
    
    % 2. Adaptive scaling factor based on constraints
    cons_max = max(cons);
    cons_min = min(cons);
    epsilon = 1e-6;
    F = 0.5 * (1 + (cons_max - cons) ./ (cons_max - cons_min + epsilon));
    
    % 3. Generate mutant vectors
    for i = 1:NP
        % Select distinct random indices
        idxs = randperm(NP, 4);
        while any(idxs == i)
            idxs = randperm(NP, 4);
        end
        
        % Select random elite
        elite_idx = randi(size(elites, 1));
        x_elite = elites(elite_idx, :);
        
        % Directional mutation
        mutant = popdecs(idxs(1), :) + F(i) .* (x_elite - popdecs(idxs(2), :)) + ...
                 F(i) .* (popdecs(idxs(3), :) - popdecs(idxs(4), :));
        
        % 4. Constraint-aware crossover
        CR_i = 0.9 - 0.5*(cons(i)/(cons_max + epsilon)) + 0.1*rand();
        j_rand = randi(D);
        mask = rand(1, D) < CR_i;
        mask(j_rand) = true;
        
        % Create trial vector
        trial = popdecs(i, :);
        trial(mask) = mutant(mask);
        
        % 5. Repair mechanism for infeasible solutions
        if cons(i) > 0 && rand() < 0.8
            trial = 0.5 * (x_elite + trial);
        end
        
        offspring(i, :) = trial;
    end
    
    % 6. Boundary handling with reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc645(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite and best solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % 2. Adaptive parameters
    max_cons = max(cons);
    min_cons = min(cons);
    mean_cons = mean(cons);
    F = 0.5 + 0.3 * (cons - min_cons) ./ (max_cons - min_cons + eps);
    p_elite = 0.4 * (1 - min(popfits)/(max(popfits) + eps));
    
    % 3. Generate mutant vectors
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Strategy 1: Elite-guided
        v1 = popdecs(i,:) + F(i)*(elite - popdecs(i,:)) + F(i)*(popdecs(r1,:) - popdecs(r2,:));
        
        % Strategy 2: Constraint-aware
        alpha = F(i)*(1 - cons(i)/(max_cons + eps));
        beta = F(i)*(cons(i)/(max_cons + eps));
        v2 = best + alpha*(popdecs(r1,:) - popdecs(r2,:)) + beta*(popdecs(r3,:) - popdecs(r4,:));
        
        % Strategy 3: Fitness-directed
        f_sum = popfits(r1) + popfits(r2) + eps;
        v3 = (popfits(r2)*popdecs(r1,:) + popfits(r1)*popdecs(r2,:))/f_sum + ...
             F(i)*(popdecs(r3,:) - popdecs(r4,:));
        
        % Select strategy
        if rand() < p_elite
            mutant(i,:) = v1;
        elseif cons(i) < mean_cons
            mutant(i,:) = v2;
        else
            mutant(i,:) = v3;
        end
    end
    
    % 4. Crossover with adaptive CR
    CR = 0.5 + 0.4*(1 - cons./(max_cons + eps));
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
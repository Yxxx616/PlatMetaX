% MATLAB Code
function [offspring] = updateFunc649(popdecs, popfits, cons)
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
    
    % 2. Enhanced adaptive parameters
    max_fit = max(popfits);
    min_fit = min(popfits);
    max_cons = max(cons);
    min_cons = min(cons);
    med_cons = median(cons);
    
    % Improved rank-based CR adaptation
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.5 + 0.4 * (ranks / NP).^0.5;
    
    % Dynamic F adaptation with constraint awareness
    F = 0.4 + 0.4 * (cons - min_cons) ./ (max_cons - min_cons + eps) + ...
        0.2 * (popfits - min_fit) ./ (max_fit - min_fit + eps);
    
    % 3. Vectorized mutant generation
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Strategy 1: Elite-guided
        v1 = elite + F(i)*(popdecs(r1,:) - popdecs(r2,:)) + F(i)*(popdecs(r3,:) - popdecs(r4,:));
        
        % Strategy 2: Fitness-directed
        f_sum = popfits(r1) + popfits(r2) + eps;
        v2 = (popfits(r2)*popdecs(r1,:) + popfits(r1)*popdecs(r2,:))/f_sum + F(i)*(popdecs(r3,:) - popdecs(r4,:));
        
        % Strategy 3: Constraint-aware
        v3 = popdecs(i,:) + F(i)*(best - popdecs(i,:)) + F(i)*(popdecs(r1,:) - popdecs(r2,:));
        
        % Enhanced strategy selection
        if cons(i) <= med_cons
            if rand() < 0.7
                mutant(i,:) = v1;
            else
                mutant(i,:) = v2;
            end
        else
            mutant(i,:) = v3;
        end
    end
    
    % 4. Improved crossover with adaptive CR
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Smart boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with random component
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = lb_rep(below_lb) + rand(sum(below_lb(:)),1).*(ub_rep(below_lb)-lb_rep(below_lb));
    offspring(above_ub) = lb_rep(above_ub) + rand(sum(above_ub(:)),1).*(ub_rep(above_ub)-lb_rep(above_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
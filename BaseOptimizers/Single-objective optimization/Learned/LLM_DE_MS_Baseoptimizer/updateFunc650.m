% MATLAB Code
function [offspring] = updateFunc650(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solution
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Adaptive parameters
    max_fit = max(popfits);
    min_fit = min(popfits);
    max_cons = max(cons);
    min_cons = min(cons);
    med_cons = median(cons);
    
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.1 + 0.8 * (ranks / NP).^0.5;
    
    F = 0.3 + 0.5 * (cons - min_cons) ./ (max_cons - min_cons + eps) + ...
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
        
        % Strategy 3: Constraint-repair
        v3 = popdecs(i,:) + F(i)*(elite - popdecs(i,:)) + F(i)*(popdecs(r1,:) - popdecs(r2,:));
        
        % Strategy selection
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
    
    % 4. Crossover with adaptive CR
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = lb_rep(below_lb) + rand(sum(below_lb(:)),1).*(ub_rep(below_lb)-lb_rep(below_lb));
    offspring(above_ub) = lb_rep(above_ub) + rand(sum(above_ub(:)),1).*(ub_rep(above_ub)-lb_rep(above_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
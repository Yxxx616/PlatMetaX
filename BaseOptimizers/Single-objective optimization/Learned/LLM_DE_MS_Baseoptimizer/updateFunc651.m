% MATLAB Code
function [offspring] = updateFunc651(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solution considering both fitness and constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, min_cons_idx] = min(cons);
        [~, min_fit_idx] = min(popfits);
        if cons(min_fit_idx) < median(cons)
            elite = popdecs(min_fit_idx, :);
        else
            elite = popdecs(min_cons_idx, :);
        end
    end
    
    % 2. Adaptive parameters calculation
    max_fit = max(popfits);
    min_fit = min(popfits);
    max_cons = max(cons);
    min_cons = min(cons);
    med_cons = median(cons);
    
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.1 + 0.8 * (ranks / NP).^0.5;
    
    F = 0.3 + 0.4 * (cons - min_cons) ./ (max_cons - min_cons + eps) + ...
        0.3 * (popfits - min_fit) ./ (max_fit - min_fit + eps);
    
    % 3. Generate random indices matrix for vectorized operations
    idx_mat = zeros(NP, 4);
    for i = 1:NP
        candidates = 1:NP;
        candidates(i) = [];
        perm = randperm(length(candidates), 4);
        idx_mat(i,:) = candidates(perm);
    end
    
    % 4. Vectorized mutant generation
    mutant = zeros(NP, D);
    for i = 1:NP
        r1 = idx_mat(i,1); r2 = idx_mat(i,2); r3 = idx_mat(i,3); r4 = idx_mat(i,4);
        
        % Strategy 1: Elite-guided
        v1 = elite + F(i)*(popdecs(r1,:) - popdecs(r2,:)) + F(i)*(popdecs(r3,:) - popdecs(r4,:));
        
        % Strategy 2: Fitness-weighted
        avg_fit = mean(popfits([r1 r2]));
        w1 = 1 ./ (1 + exp(popfits(r1) - avg_fit));
        w2 = 1 ./ (1 + exp(popfits(r2) - avg_fit));
        v2 = (w1*popdecs(r1,:) + w2*popdecs(r2,:))/(w1 + w2 + eps) + F(i)*(popdecs(r3,:) - popdecs(r4,:));
        
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
    
    % 5. Crossover with adaptive CR
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement with random reinitialization if still out of bounds
    still_below = offspring < lb_rep;
    still_above = offspring > ub_rep;
    offspring(still_below) = lb_rep(still_below) + rand(sum(still_below(:)),1).*(ub_rep(still_below)-lb_rep(still_below));
    offspring(still_above) = lb_rep(still_above) + rand(sum(still_above(:)),1).*(ub_rep(still_above)-lb_rep(still_above));
end
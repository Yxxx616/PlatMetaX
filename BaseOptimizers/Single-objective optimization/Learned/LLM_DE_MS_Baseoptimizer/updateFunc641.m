% MATLAB Code
function [offspring] = updateFunc641(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
        elite_fit = popfits(temp(elite_idx));
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
        elite_fit = popfits(elite_idx);
    end
    
    % 2. Adaptive parameters
    max_cons = max(cons);
    min_cons = min(cons);
    max_fit = max(popfits);
    
    F = 0.4 + 0.4 * (cons - min_cons) ./ (max_cons - min_cons + eps);
    p_elite = 0.3 * (1 - elite_fit / (max_fit + eps));
    mean_cons = mean(cons);
    
    % 3. Multi-strategy mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 6));
        
        % Elite-guided direction
        v1 = popdecs(i,:) + F(i) * (elite - popdecs(i,:));
        
        % Constraint-aware differential
        scale = 1 - cons(i)/(max_cons + eps);
        v2 = popdecs(idx(1),:) + F(i) * (popdecs(idx(2),:) - popdecs(idx(3),:)) * scale;
        
        % Fitness-weighted recombination
        r = idx(4:6);
        f_r = popfits(r);
        weights = exp(-f_r) / sum(exp(-f_r));
        v3 = weights(1)*popdecs(r(1),:) + weights(2)*popdecs(r(2),:) + weights(3)*popdecs(r(3),:);
        
        % Strategy selection
        if rand() < p_elite
            mutant(i,:) = v1;
        elseif cons(i) < mean_cons
            mutant(i,:) = v2;
        else
            mutant(i,:) = v3;
        end
    end
    
    % 4. Adaptive crossover
    CR = 0.9 * (1 - cons./(max_cons + eps));
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
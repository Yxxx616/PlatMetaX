% MATLAB Code
function [offspring] = updateFunc639(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Adaptive scaling factor
    mean_cons = mean(cons);
    std_cons = std(cons);
    if std_cons == 0, std_cons = 1; end
    F = 0.5 + 0.3 * tanh((mean_cons - cons)./(std_cons + eps));
    
    % 3. Hybrid mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        
        diff_vec = (popdecs(idx(1),:) - popdecs(idx(2),:) + ...
                  popdecs(idx(3),:) - popdecs(idx(4),:)) / 2;
        
        mutant(i,:) = popdecs(i,:) + F(i)*(elite - popdecs(i,:)) + F(i)*diff_vec;
    end
    
    % 4. Constraint-aware crossover
    max_cons = max(cons);
    if max_cons == 0, max_cons = 1; end
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
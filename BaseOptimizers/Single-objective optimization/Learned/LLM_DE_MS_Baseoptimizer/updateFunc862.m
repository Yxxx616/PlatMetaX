% MATLAB Code
function [offspring] = updateFunc862(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify best solution (feasible first)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(cons);
        best = popdecs(best_idx, :);
    end
    
    % Adaptive scaling factor based on constraints
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    F = 0.5 + 0.2 * tanh((cons - mean_c) ./ std_c);
    
    % Generate mutation indices (ensure distinct)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    for i = 1:NP
        while r1(i) == i || r2(i) == i || r3(i) == i || ...
              r1(i) == r2(i) || r1(i) == r3(i) || r2(i) == r3(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
            r3(i) = randi(NP);
        end
    end
    
    % Novel mutation strategy
    best_diff = best - popdecs;
    avg_diff = (popdecs(r1,:) + popdecs(r2,:))/2 - popdecs(r3,:);
    mutant = popdecs + F.*best_diff + (1-F).*avg_diff;
    
    % Rank-based crossover rate
    [~, rank] = sort(popfits);
    CR = 0.85 - 0.35 * (rank-1)/NP;
    
    % Binomial crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Random dimension reset for infeasible solutions
    reset_mask = cons > 0 & rand(NP,1) < 0.1;
    if any(reset_mask)
        dims = randi(D, sum(reset_mask), 1);
        idx = find(reset_mask);
        for i = 1:length(idx)
            offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
        end
    end
end
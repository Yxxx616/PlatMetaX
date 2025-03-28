% MATLAB Code
function [offspring] = updateFunc863(popdecs, popfits, cons)
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
    F = 0.5 + 0.3 * tanh((cons - mean_c) ./ std_c);
    
    % Generate mutation indices (ensure distinct)
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    r3 = zeros(NP, 1);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 3));
        r1(i) = selected(1);
        r2(i) = selected(2);
        r3(i) = selected(3);
    end
    
    % Enhanced mutation strategy
    best_diff = best - popdecs;
    avg_diff = (popdecs(r1,:) + popdecs(r2,:))/2 - popdecs(r3,:);
    rand_perturb = 0.2 * (rand(NP,D)-0.5) .* (ub-lb);
    mutant = popdecs + F.*best_diff + (1-F).*avg_diff + rand_perturb;
    
    % Dynamic crossover rate based on fitness rank
    [~, rank] = sort(popfits);
    CR = 0.9 - 0.5 * (rank-1)/NP;
    
    % Binomial crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection and random reset
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for boundary violations
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Enhanced diversity mechanism for infeasible solutions
    reset_prob = 0.15 + 0.35 * (cons - min(cons))/(max(cons)-min(cons)+eps);
    reset_mask = rand(NP,1) < reset_prob;
    if any(reset_mask)
        dims = randi(D, sum(reset_mask), 1);
        idx = find(reset_mask);
        for i = 1:length(idx)
            offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
        end
    end
end
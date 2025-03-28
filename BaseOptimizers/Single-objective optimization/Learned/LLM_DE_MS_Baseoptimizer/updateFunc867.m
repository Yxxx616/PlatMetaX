% MATLAB Code
function [offspring] = updateFunc867(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility and fitness information
    feasible = cons <= 0;
    [~, sorted_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(sorted_idx) = 1:NP;
    
    % Best and worst individuals
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(popfits + cons);
        best = popdecs(best_idx, :);
    end
    [~, worst_idx] = max(popfits + cons);
    
    % Adaptive scaling factor
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    F = 0.5 * (1 + tanh((cons - mean_c)./std_c));
    F = min(max(F, 0.1), 0.9);
    
    % Rank-based crossover probability
    CR = 0.9 - 0.4 * (rank-1)/NP;
    
    % Mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select distinct indices
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 4));
        r1 = selected(1); r2 = selected(2);
        r3 = selected(3); r4 = selected(4);
        
        % Base vector selection
        if any(feasible)
            base = best;
        else
            norm_fit = (popfits(i) - popfits(worst_idx))/(popfits(best_idx) - popfits(worst_idx) + eps);
            base = popdecs(i,:) + norm_fit * (best - popdecs(i,:));
        end
        
        % Constraint-aware mutation
        mutant(i,:) = base + F(i) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                     (1-F(i)) * (cons(i)/(max(cons)+eps)) * (popdecs(r3,:) - popdecs(r4,:));
    end
    
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
    
    % Constraint-based diversity maintenance
    reset_prob = 0.2 * (cons - min(cons))/(max(cons) - min(cons) + eps);
    reset_mask = rand(NP,1) < reset_prob;
    if any(reset_mask)
        dims = randi(D, sum(reset_mask), 1);
        idx = find(reset_mask);
        for i = 1:length(idx)
            offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
        end
    end
end
% MATLAB Code
function [offspring] = updateFunc865(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Best selection considering both fitness and constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(cons + popfits); % Combined metric
        best = popdecs(best_idx, :);
    end
    
    % Adaptive scaling factor
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    mean_f = mean(popfits);
    std_f = std(popfits) + eps;
    
    sigmoid = @(x) 1./(1 + exp(-x));
    F = 0.4 + 0.3 * sigmoid((cons - mean_c)./std_c) + ...
        0.2 * (popfits - mean_f)./std_f;
    F = min(max(F, 0.1), 0.9);
    
    % Mutation indices (distinct)
    r1 = zeros(NP, 1); r2 = zeros(NP, 1);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 2));
        r1(i) = selected(1);
        r2(i) = selected(2);
    end
    
    % Directional mutation
    best_diff = best - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    cons_weight = 0.1 * (1 + cons./(max(cons) + eps));
    rand_perturb = bsxfun(@times, randn(NP,D), cons_weight) .* (ub-lb);
    
    mutant = popdecs + bsxfun(@times, F, best_diff) + ...
             bsxfun(@times, 1-F, rand_diff) + rand_perturb;
    
    % Rank-based crossover
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
    
    % Constraint-aware diversity
    reset_prob = 0.25 * cons ./ (max(cons) + eps);
    reset_mask = rand(NP,1) < reset_prob;
    if any(reset_mask)
        dims = randi(D, sum(reset_mask), 1);
        idx = find(reset_mask);
        for i = 1:length(idx)
            offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
        end
    end
end
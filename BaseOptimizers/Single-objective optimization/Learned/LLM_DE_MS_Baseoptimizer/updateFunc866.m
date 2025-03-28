% MATLAB Code
function [offspring] = updateFunc866(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility-aware best selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(popfits + cons);
        best = popdecs(best_idx, :);
    end
    
    % Adaptive scaling factor with constraint information
    mean_c = mean(cons);
    std_c = std(cons) + eps;
    F = 0.5 * (1 + tanh((cons - mean_c)./std_c));
    F = min(max(F, 0.1), 0.9);
    
    % Mutation with distinct indices
    mutant = zeros(NP, D);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 2));
        r1 = selected(1);
        r2 = selected(2);
        
        % Directional mutation
        mutant(i,:) = popdecs(i,:) + F(i) * (best - popdecs(i,:)) + ...
                     (1-F(i)) * (popdecs(r1,:) - popdecs(r2,:));
        
        % Constraint-driven perturbation
        perturb = randn(1,D) .* (cons(i)/(max(cons)+eps)) .* (ub-lb);
        mutant(i,:) = mutant(i,:) + perturb;
    end
    
    % Rank-based crossover probability
    [~, rank] = sort(popfits);
    CR = 0.9 - 0.5 * (rank-1)/NP;
    
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
    
    % Constraint-based diversity enhancement
    reset_prob = 0.3 * cons ./ (max(cons) + eps);
    reset_mask = rand(NP,1) < reset_prob;
    if any(reset_mask)
        dims = randi(D, sum(reset_mask), 1);
        idx = find(reset_mask);
        for i = 1:length(idx)
            offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
        end
    end
end
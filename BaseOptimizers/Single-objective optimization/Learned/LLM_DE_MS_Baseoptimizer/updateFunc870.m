% MATLAB Code
function [offspring] = updateFunc870(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible = cons <= 0;
    num_feasible = sum(feasible);
    
    % Best individual selection
    if num_feasible > 0
        [~, best_idx] = min(popfits(feasible));
        best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(popfits);
        best = popdecs(best_idx, :);
    end
    
    % Normalized fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive parameters
    sigma_f = std(popfits) + eps;
    sigma_c = std(cons) + eps;
    F = 0.5 + 0.4 * tanh(cons/sigma_c);
    
    % Rank-based weights
    [~, sorted_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(sorted_idx) = 1:NP;
    lambda = 0.5 * (1 - tanh(rank/NP));
    
    % Mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select base vectors
        if num_feasible > 0
            candidates = find(feasible);
        else
            candidates = 1:NP;
        end
        candidates = setdiff(candidates, i);
        selected = candidates(randperm(length(candidates), 3));
        
        % Weighted difference
        r1 = selected(1); r2 = selected(2); r3 = selected(3);
        mutant(i,:) = popdecs(r1,:) + F(i)*(popdecs(r2,:)-popdecs(r3,:)) + ...
                     lambda(i)*(best - popdecs(i,:));
    end
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * norm_cons;
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
    
    % Constraint-based diversity
    if max(cons) > 0
        reset_prob = 0.1 * norm_cons;
        reset_mask = rand(NP,1) < reset_prob;
        if any(reset_mask)
            dims = randi(D, sum(reset_mask), 1);
            idx = find(reset_mask);
            for i = 1:length(idx)
                offspring(idx(i), dims(i)) = lb(dims(i)) + rand()*(ub(dims(i))-lb(dims(i)));
            end
        end
    end
end
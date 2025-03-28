% MATLAB Code
function [offspring] = updateFunc583(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    cons_norm = cons_pos ./ (max(cons_pos) + eps);
    
    % Elite selection - best feasible or least infeasible
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*cons_pos); % Heavy penalty for infeasible
        elite = popdecs(elite_idx, :);
    end
    
    % Rank population (1=best, NP=worst) considering both fitness and constraints
    [~, rank_order] = sort(popfits + 1e3*cons_pos);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    rank_weights = 1 - (ranks-1)/(NP-1); % Normalized weights [1,0]
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive scaling factors
    F = 0.3 + 0.5 * rank_weights .* (1 - cons_norm);
    
    % Constraint-aware mutation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    cons_weight = (1 - cons_norm);
    mutant = popdecs + F .* elite_diff + F .* rand_diff .* cons_weight(:, ones(1,D));
    
    % Adaptive crossover
    CR = 0.1 + 0.7 * rank_weights .* (1 - cons_norm);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | (1:D) == j_rand(:, ones(1,D));
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - bounce back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    offspring(lower_violation) = lb_rep(lower_violation) + 0.5*(popdecs(lower_violation) - lb_rep(lower_violation));
    offspring(upper_violation) = ub_rep(upper_violation) - 0.5*(ub_rep(upper_violation) - popdecs(upper_violation));
    
    % Final clipping to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
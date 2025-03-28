% MATLAB Code
function [offspring] = updateFunc582(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    cons_norm = cons_pos ./ (max(cons_pos) + eps);
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 10*cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Rank population (1=best, NP=worst)
    [~, rank_order] = sort(popfits + 10*cons_pos);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    rank_weights = 1 - (ranks-1)/(NP-1);
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive scaling factors
    F = 0.4 + 0.4 * rank_weights .* (1 - cons_norm);
    
    % Directional mutation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    mutant = popdecs + F .* elite_diff + F .* rand_diff;
    
    % Adaptive crossover
    CR = 0.2 + 0.6 * rank_weights .* (1 - cons_norm);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | (1:D) == j_rand(:, ones(1,D));
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    offspring(lower_violation) = (offspring(lower_violation) + lb_rep(lower_violation))/2;
    offspring(upper_violation) = (offspring(upper_violation) + ub_rep(upper_violation))/2;
    
    % Final clipping to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
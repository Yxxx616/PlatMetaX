% MATLAB Code
function [offspring] = updateFunc579(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    cons_norm = (cons_pos - min(cons_pos)) ./ (max(cons_pos) - min(cons_pos) + eps);
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Rank population (1=best, NP=worst)
    [~, rank_order] = sort(popfits + cons_pos);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    rank_norm = (ranks-1)/(NP-1);  % Normalized to [0,1]
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    w1 = 0.8 * (1 - rank_norm);
    w2 = 0.2 * rank_norm;
    F = 0.5 + 0.4 * (1 - cons_norm) .* cos(pi * rank_norm);
    sigma = 0.1 * (1 + cons_norm);
    
    % Mutation with directional guidance
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = sigma .* randn(NP, D);
    mutant = popdecs + F .* (w1.*elite_diff + w2.*rand_diff) + noise;
    
    % Adaptive crossover
    CR = 0.2 + 0.6 * (1 - rank_norm) .* (1 - cons_norm);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | (1:D) == j_rand;
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - reflection with damping
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    offspring(lower_violation) = lb_rep(lower_violation) + 0.5*(offspring(lower_violation) - lb_rep(lower_violation));
    offspring(upper_violation) = ub_rep(upper_violation) - 0.5*(offspring(upper_violation) - ub_rep(upper_violation));
    
    % Final clipping to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc588(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    cons_norm = cons_pos ./ (max(cons_pos) + eps);
    
    % Elite selection with feasibility rules
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Rank population considering both fitness and constraints
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(cons_pos);
    rank_weights = (1:NP)'/NP; % Normalized weights [1/NP, 1]
    
    % Generate 4 distinct random indices for each individual
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    r3 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], [r1(i) r2(i)])(randi(NP-3)), idx)';
    r4 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], [r1(i) r2(i) r3(i)])(randi(NP-4)), idx)';
    
    % Adaptive scaling factors
    F = 0.5 + 0.3*(1 - cons_norm) + 0.2*(1 - rank_weights);
    
    % Mutation with elite guidance
    elite_rep = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutant = elite_rep + F(:, ones(1,D)) .* diff1 + 0.5 * F(:, ones(1,D)) .* diff2;
    
    % Adaptive crossover
    CR = 0.3 + 0.5 * (1 - cons_norm) .* (1 - rank_weights);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | (1:D) == j_rand(:, ones(1,D));
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Ensure final bounds
    offspring = min(max(offspring, lb_rep), ub_rep);
end
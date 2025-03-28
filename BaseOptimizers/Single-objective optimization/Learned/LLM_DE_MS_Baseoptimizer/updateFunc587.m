% MATLAB Code
function [offspring] = updateFunc587(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    cons_norm = cons_pos ./ (max(cons_pos) + eps;
    
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
    
    % Generate 6 distinct random indices for each individual
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    r3 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], [r1(i) r2(i)])(randi(NP-3)), idx)';
    r4 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], [r1(i) r2(i) r3(i)])(randi(NP-4)), idx)';
    r5 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], [r1(i) r2(i) r3(i) r4(i)])(randi(NP-5)), idx)';
    r6 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], [r1(i) r2(i) r3(i) r4(i) r5(i)])(randi(NP-6)), idx)';
    
    % Multi-component mutation
    elite_rep = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    diff3 = popdecs(r5,:) - popdecs(r6,:);
    
    mutant = elite_rep + 0.5 * diff1 + ...
             0.3 * (1 - cons_norm(:, ones(1,D))) .* diff2 + ...
             0.2 * (1 - rank_weights(:, ones(1,D))) .* diff3;
    
    % Adaptive crossover
    CR = 0.1 + 0.7 * (1 - cons_norm) .* (1 - rank_weights);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | (1:D) == j_rand(:, ones(1,D));
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - random reinitialization
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_values = lb_rep + (ub_rep - lb_rep) .* rand(NP, D);
    offspring(out_of_bounds) = rand_values(out_of_bounds);
end
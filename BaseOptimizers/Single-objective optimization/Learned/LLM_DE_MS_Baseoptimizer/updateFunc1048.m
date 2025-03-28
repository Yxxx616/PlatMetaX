% MATLAB Code
function [offspring] = updateFunc1048(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    norm_f = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_c = abs(cons) / (max(abs(cons)) + eps);
    
    % Select top 30% as elites
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.3*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    best = elite_pool(1,:);
    
    % Calculate weights (non-linear ranking)
    ranks = 1:elite_num;
    weights = (elite_num - ranks + 1).^2 / sum((1:elite_num).^2);
    
    % Elite direction component (vectorized)
    elite_diff = reshape(elite_pool, elite_num, 1, D) - reshape(popdecs, 1, NP, D);
    weighted_diff = sum(reshape(weights, elite_num, 1, 1) .* elite_diff, 1);
    d_elite = squeeze(weighted_diff)';
    
    % Constraint-aware perturbation
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    sigma = 1 + tanh(norm_c);
    d_cons = sigma .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Random exploration component
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    d_rand = bsxfun(@minus, best, popdecs) + (popdecs(r3,:) - popdecs(r4,:));
    
    % Adaptive scaling factors
    F_base = 0.5 + 0.3 * sin(pi * norm_f / 2);
    alpha = 0.3 * (1 - norm_c);
    beta = 0.2 * norm_f;
    
    % Combined mutation
    mutants = popdecs + F_base.*d_elite + alpha.*d_cons + beta.*d_rand;
    
    % Dynamic crossover
    [~, ranks] = sort(popfits);
    norm_rank = (ranks-1)/(NP-1);
    CR = 0.85 * (1 - norm_rank.^0.5);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling - reflection with random component
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = lb(lb_viol) + rand(sum(lb_viol(:)),1).*(ub(lb_viol)-lb(lb_viol));
    offspring(ub_viol) = lb(ub_viol) + rand(sum(ub_viol(:)),1).*(ub(ub_viol)-lb(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
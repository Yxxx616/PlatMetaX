% MATLAB Code
function [offspring] = updateFunc1041(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Sort population by fitness (lower is better)
    [~, sorted_idx] = sort(popfits);
    
    % Elite selection (top 20%)
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Compute weighted centroid of elites
    weights = 1./(1 + log(1 + (1:elite_num)'));
    centroid = sum(weights .* elite_pool, 1) / sum(weights);
    
    % Normalize constraint violations
    min_c = min(cons);
    max_c = max(cons);
    norm_cons = (cons - min_c) / (max_c - min_c + eps);
    
    % Adaptive scaling factors based on constraints
    F = 0.4 + 0.3 * (1 - norm_cons);
    
    % Generate random elite indices for mutation
    e1 = randi(elite_num, NP, 1);
    e2 = mod(e1 + randi(elite_num-1, NP, 1) + 1;
    
    % Elite-guided mutation with directional component
    mutants = popdecs + F.*(elite_pool(e1,:) - elite_pool(e2,:)) + 0.5*(centroid - popdecs);
    
    % Adaptive crossover rates based on rank
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    CR = 0.9 - 0.5 * (ranks/NP);
    
    % Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection or reinitialization
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    reflect = rand(NP,D) < 0.5;
    
    % Reflection
    offspring(lb_viol & reflect) = 2*lb(lb_viol & reflect) - offspring(lb_viol & reflect);
    offspring(ub_viol & reflect) = 2*ub(ub_viol & reflect) - offspring(ub_viol & reflect);
    
    % Reinitialization
    rand_mask = (lb_viol | ub_viol) & ~reflect;
    offspring(rand_mask) = lb(rand_mask) + (ub(rand_mask)-lb(rand_mask)).*rand(sum(rand_mask(:)),1);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc1590(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Centroid of top 30% solutions
    [~, sorted_idx] = sort(popfits);
    top30 = sorted_idx(1:ceil(NP*0.3));
    centroid = mean(popdecs(top30, :), 1);
    
    % 3. Adaptive weights based on constraint violation
    norm_cons = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    alpha = 0.5 - 0.3*norm_cons;
    beta = 0.3 + 0.2*norm_cons;
    gamma = 0.2 + 0.3*rand(NP, 1);
    
    % 4. Rank-based scaling factors
    [~, ranks] = sort(popfits);
    F = 0.4 + 0.4*(1 - ranks/NP);
    
    % 5. Select random vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    same_idx = rand_idx1 == rand_idx2;
    rand_idx2(same_idx) = mod(rand_idx2(same_idx) + randi(NP-1, sum(same_idx), 1), NP) + 1;
    
    % 6. Personal best selection (top 30%)
    pbest_pool = popdecs(top30, :);
    pbest_idx = randi(length(top30), NP, 1);
    pbest = pbest_pool(pbest_idx, :);
    
    % 7. Generate direction vectors (vectorized)
    dir_vectors = alpha.*(elite - popdecs) + ...
                 beta.*(centroid - popdecs) + ...
                 gamma.*(popdecs(rand_idx1,:) - popdecs(rand_idx2,:));
    
    % 8. Generate mutation vectors
    eta = 0.1 * rand(NP, 1);
    offspring = popdecs + F.*dir_vectors + eta.*(pbest - popdecs);
    
    % 9. Rank-based crossover
    CR = 0.8 + 0.1*(1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 10. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % 11. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
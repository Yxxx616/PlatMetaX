% MATLAB Code
function [offspring] = updateFunc1589(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solution (considering constraints)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute population centroid
    centroid = mean(popdecs, 1);
    
    % 3. Adaptive weights based on constraint violation
    norm_cons = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    alpha = 0.6 - 0.3*norm_cons;
    beta = 0.3 + 0.2*norm_cons;
    gamma = 0.1 + 0.2*rand(NP, 1);
    
    % 4. Rank-based scaling factors
    [~, ranks] = sort(popfits);
    F = 0.5 + 0.3*(1 - ranks/NP);
    
    % 5. Select random vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    same_idx = rand_idx1 == rand_idx2;
    rand_idx2(same_idx) = mod(rand_idx2(same_idx) + randi(NP-1, sum(same_idx), 1), NP) + 1;
    
    % 6. Personal best selection (top 30%)
    [~, sorted_idx] = sort(popfits);
    pbest_pool = popdecs(sorted_idx(1:ceil(NP*0.3)), :);
    pbest_idx = randi(size(pbest_pool, 1), NP, 1);
    pbest = pbest_pool(pbest_idx, :);
    
    % 7. Generate direction vectors
    dir_vectors = zeros(NP, D);
    for i = 1:NP
        dir_vectors(i,:) = alpha(i)*(elite - popdecs(i,:)) + ...
                          beta(i)*(centroid - popdecs(i,:)) + ...
                          gamma(i)*(popdecs(rand_idx1(i),:) - popdecs(rand_idx2(i),:));
    end
    
    % 8. Generate mutation vectors
    eta = 0.05 * rand(NP, 1);
    offspring = popdecs + F.*dir_vectors + eta.*(pbest - popdecs);
    
    % 9. Rank-based crossover
    CR = 0.7 + 0.2*(1 - ranks/NP);
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
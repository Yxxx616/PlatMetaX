% MATLAB Code
function [offspring] = updateFunc683(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute weights combining fitness and constraints
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons)/max_cons;
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    weights = (1 - norm_fits) .* (1 - norm_cons);
    weights = weights / sum(weights);
    
    % 3. Generate random indices avoiding current index
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx == (1:NP)';
    rand_idx = rand_idx + cumsum(mask, 2);
    rand_idx = mod(rand_idx-1, NP) + 1;
    r1 = rand_idx(:,1);
    r2 = rand_idx(:,2);
    
    % 4. Compute direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Weighted centroid direction
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, popdecs - popdecs(i,:), weights), 1);
    end
    
    % 5. Adaptive scaling factors
    F1 = 0.5 + 0.5*randn(NP, 1);
    F2 = 0.3 * (1 - norm_fits);
    F3 = 0.2 * (1 - norm_cons);
    
    % 6. Mutation operation
    mutant = popdecs + F1.*elite_dir + F2.*rand_dir + F3.*weighted_diff;
    
    % 7. Rank-based adaptive crossover
    [~, rank_order] = sort(popfits + 1e6*max(0,cons));
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.9 * (1 - ranks/NP);
    
    mask = rand(NP, D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
end
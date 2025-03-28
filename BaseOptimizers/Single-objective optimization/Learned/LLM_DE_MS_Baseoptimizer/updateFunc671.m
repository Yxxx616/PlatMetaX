% MATLAB Code
function [offspring] = updateFunc671(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with improved constraint handling
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Enhanced ranking considering both fitness and constraints
    penalty = 1e6 * max(0, cons);
    combined = popfits + penalty;
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    % 3. Generate random indices (vectorized and improved)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx == (1:NP)';
    rand_idx = rand_idx + cumsum(mask, 2);
    rand_idx = mod(rand_idx-1, NP) + 1;
    r1 = rand_idx(:,1); r2 = rand_idx(:,2);
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % 4. Compute direction vectors with improved weighting
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    
    % Improved constraint-aware direction
    tau = max(5, 0.15*NP);
    weights = exp(-ranks/tau);
    weights = weights / sum(weights);
    weighted_diff = popdecs' - reshape(popdecs, 1, NP, D);
    weighted_diff = squeeze(sum(weighted_diff .* reshape(weights, 1, NP, 1), 2))';
    cons_dir = weighted_diff .* (1 + norm_cons);
    
    % 5. Refined adaptive scaling factors
    F_elite = 0.8 * (1 - norm_ranks);
    F_rand = 0.4 + 0.2 * randn(NP,1);
    F_cons = 0.2 * (1 - norm_cons);
    
    % 6. Mutation operation with better balance
    mutant = popdecs + F_elite.*elite_dir + F_rand.*rand_dir + F_cons.*cons_dir;
    
    % 7. Improved crossover with wider range
    CR = 0.1 + 0.8 * (1 - norm_ranks);
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 8. Enhanced boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
end
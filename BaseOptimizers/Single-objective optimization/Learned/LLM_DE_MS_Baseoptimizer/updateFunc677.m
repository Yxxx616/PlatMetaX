% MATLAB Code
function [offspring] = updateFunc677(popdecs, popfits, cons)
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
    
    % 3. Constraint normalization
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    % 4. Generate random indices (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx == (1:NP)';
    rand_idx = rand_idx + cumsum(mask, 2);
    rand_idx = mod(rand_idx-1, NP) + 1;
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2);
    
    % 5. Compute direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rank_dir = (popdecs(r1,:) - popdecs(r2,:)) .* (1 - norm_ranks);
    
    % Constraint-aware direction with exponential weights
    tau = max(5, 0.2*NP);
    weights = exp(-ranks/tau);
    weights = weights / sum(weights);
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, popdecs - popdecs(i,:), weights), 1);
    end
    cons_dir = weighted_diff .* (1 + norm_cons);
    
    % Random perturbation
    rand_dir = (rand(NP, D)*2 - 1) .* (ub-lb)/5;
    
    % 6. Adaptive scaling factors (improved)
    F_elite = 0.8 + 0.2 * randn(NP,1);
    F_rank = 0.6 * (1 - norm_ranks);
    F_cons = 0.4 * (1 - norm_cons);
    
    % 7. Mutation operation
    mutant = popdecs + F_elite.*elite_dir + F_rank.*rank_dir + ...
             F_cons.*cons_dir + 0.2.*rand_dir;
    
    % 8. Dynamic crossover
    CR = 0.9 - 0.3 * norm_ranks;
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
end
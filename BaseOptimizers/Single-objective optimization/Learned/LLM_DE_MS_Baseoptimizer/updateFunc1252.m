% MATLAB Code
function [offspring] = updateFunc1252(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        x_elite = popdecs(elite_idx, :);
    end
    
    % 2. Select top 3 solutions based on fitness
    [~, sorted_idx] = sort(popfits);
    top3 = sorted_idx(1:3);
    
    % 3. Compute fitness weights
    f_vals = popfits(top3);
    weights = exp(-(f_vals - min(f_vals)));
    weights = weights / sum(weights);
    
    % 4. Generate direction vectors
    d_elite = x_elite(ones(NP,1), :) - popdecs;
    
    % Fitness-weighted difference
    rand_idx = randi(NP, NP, 3);
    d_fit = weights(1)*(popdecs(top3(1),:) - popdecs(rand_idx(:,1),:)) + ...
            weights(2)*(popdecs(top3(2),:) - popdecs(rand_idx(:,2),:)) + ...
            weights(3)*(popdecs(top3(3),:) - popdecs(rand_idx(:,3),:));
    
    % Constraint-aware perturbation
    R = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, [i, elite_idx]);
        R(i,:) = available(randperm(length(available), 2));
    end
    d_cons = sign(cons) .* (popdecs(R(:,1),:) - popdecs(R(:,2),:));
    
    % 5. Adaptive scaling factors
    F_base = 0.5 + 0.5./(1 + exp(-abs(cons)/100));
    F = F_base .* (1 + 0.1*randn(NP,1));
    F = min(max(F, 0.1), 1.0);
    F = F(:, ones(1, D));
    
    % 6. Mutation with weighted directions
    mutants = popdecs + F .* (0.6*d_elite + 0.3*d_fit + 0.1*d_cons);
    
    % 7. Rank-based adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 - 0.5 * (ranks/NP);
    CR = CR(:, ones(1, D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end
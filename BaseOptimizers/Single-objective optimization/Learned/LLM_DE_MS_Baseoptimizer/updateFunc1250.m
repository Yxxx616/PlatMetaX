% MATLAB Code
function [offspring] = updateFunc1250(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    penalty = popfits + 1000 * max(0, cons);
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate random indices for difference vectors
    idx = 1:NP;
    R = zeros(NP, 3);
    for i = 1:NP
        available = idx(idx ~= i & idx ~= elite_idx);
        R(i,:) = available(randperm(length(available), 3));
    end
    
    % 3. Select top 3 solutions based on fitness
    [~, sorted_idx] = sort(popfits);
    top3 = sorted_idx(1:3);
    
    % 4. Compute fitness weights
    f_vals = popfits(top3);
    weights = exp(-f_vals - min(-f_vals));
    weights = weights / sum(weights);
    
    % 5. Compute direction vectors
    d_elite = x_elite(ones(NP,1), :) - popdecs;
    
    % Fitness-weighted difference
    rand_idx = randi(NP, NP, 3);
    d_fit = weights(1)*(popdecs(top3(1),:) - popdecs(rand_idx(:,1),:)) + ...
            weights(2)*(popdecs(top3(2),:) - popdecs(rand_idx(:,2),:)) + ...
            weights(3)*(popdecs(top3(3),:) - popdecs(rand_idx(:,3),:));
    
    % Constraint-aware perturbation
    d_cons = sign(cons) .* (popdecs(R(:,1),:) - popdecs(R(:,2),:));
    
    % 6. Adaptive scaling factors
    F_base = 0.5 + 0.5./(1 + exp(-abs(cons)));
    F = F_base .* (1 + 0.1*randn(NP,1));
    F = min(max(F, 0.1), 1.0);
    F = F(:, ones(1, D));
    
    % 7. Mutation with weighted directions
    mutants = popdecs + F .* (0.6*d_elite + 0.3*d_fit + 0.1*d_cons);
    
    % 8. Rank-based adaptive crossover
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
    
    % 9. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end
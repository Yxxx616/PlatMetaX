% MATLAB Code
function [offspring] = updateFunc1289(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        x_elite = popdecs(elite_idx, :);
    end
    
    % 2. Select top 3 solutions by fitness
    [~, sorted_idx] = sort(popfits);
    top3 = popdecs(sorted_idx(1:3), :);
    
    % 3. Generate direction vectors
    % Elite direction
    d_elite = x_elite - popdecs;
    
    % Random difference vectors
    R = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        R(i,:) = available(randperm(length(available), 2));
    end
    d_rand = popdecs(R(:,1),:) - popdecs(R(:,2),:);
    
    % Fitness-weighted difference (top3 influence)
    weights = [0.5, 0.3, 0.2];
    d_fit = zeros(NP, D);
    for k = 1:3
        d_fit = d_fit + weights(k) * (top3(k,:) - popdecs);
    end
    
    % 4. Adaptive scaling factors
    max_cons = max(abs(cons));
    F = 0.5 * (1 + tanh(abs(cons)/max_cons));
    F = F(:, ones(1,D));
    
    % Dynamic weights
    w_elite = 0.6;
    w_rand = 0.3;
    
    % 5. Mutation
    mutants = popdecs + F .* (w_elite*d_elite + w_rand*d_rand + (1-w_elite-w_rand)*d_fit);
    
    % 6. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 * (1 - ranks/NP);
    CR = CR(:, ones(1,D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with bounce-back
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + rand(sum(lb_mask(:)),1).*(popdecs(lb_mask)-lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - rand(sum(ub_mask(:)),1).*(ub(ub_mask)-popdecs(ub_mask));
end
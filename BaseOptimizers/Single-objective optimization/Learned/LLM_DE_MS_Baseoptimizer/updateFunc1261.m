% MATLAB Code
function [offspring] = updateFunc1261(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        x_elite = popdecs(elite_idx, :);
    end
    
    % 2. Select top 3 solutions
    [~, sorted_idx] = sort(popfits);
    top3 = popdecs(sorted_idx(1:3), :);
    
    % 3. Compute fitness weights using sigmoid
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    weights = 1 ./ (1 + exp(-norm_fits(sorted_idx(1:3))));
    weights = weights / sum(weights);
    
    % 4. Generate direction vectors
    % Elite direction
    d_elite = x_elite - popdecs;
    
    % Fitness-weighted difference from top3
    rand_idx = randi(NP, NP, 3);
    d_fit = zeros(NP, D);
    for k = 1:3
        d_fit = d_fit + weights(k) * (top3(k,:) - popdecs(rand_idx(:,k),:));
    end
    
    % Constraint-aware random difference
    R = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        R(i,:) = available(randperm(length(available), 2));
    end
    cons_factor = 1 + tanh(cons);
    d_cons = (popdecs(R(:,1),:) - popdecs(R(:,2),:)) .* cons_factor(:, ones(1, D));
    
    % 5. Adaptive scaling factors
    F = 0.5 + 0.4 ./ (1 + exp(-norm_fits));
    F = F(:, ones(1, D));
    
    % 6. Mutation with weighted directions
    mutants = popdecs + F .* (0.5*d_elite + 0.3*d_fit + 0.2*d_cons);
    
    % 7. Rank-based adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.85 - 0.35 * (ranks/NP).^1.2;
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
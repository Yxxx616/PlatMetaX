% MATLAB Code
function [offspring] = updateFunc1246(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    penalty = popfits + 1000 * max(0, cons);
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate random indices for difference vectors
    idx = 1:NP;
    R = zeros(NP, 4);
    for i = 1:NP
        available = idx(idx ~= i & idx ~= elite_idx);
        R(i,:) = available(randperm(length(available), 4));
    end
    
    % Compute fitness weights for difference vectors
    f_vals = popfits(R(:,1:2));
    weights = exp(-f_vals);
    weights = weights ./ sum(weights, 2);
    
    % Compute direction vectors
    d_elite = x_elite(ones(NP,1), :) - popdecs;
    d_fit = weights(:,1) .* (popdecs(R(:,1),:) - popdecs(R(:,2),:)) + ...
            weights(:,2) .* (popdecs(R(:,1),:) - popdecs(R(:,3),:));
    d_cons = tanh(abs(cons)) .* (popdecs(R(:,4),:) - popdecs(R(:,2),:));
    
    % 3. Compute adaptive scaling factors
    F = 0.5 + 0.3 * tanh(abs(cons)) .* randn(NP, 1);
    F = F(:, ones(1, D));
    
    % 4. Mutation with weighted directions
    mutants = popdecs + F .* (0.6*d_elite + 0.3*d_fit + 0.1*d_cons);
    
    % 5. Rank-based adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 - 0.6 * (ranks/NP);
    CR = CR(:, ones(1, D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end
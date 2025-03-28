% MATLAB Code
function [offspring] = updateFunc1241(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    penalty = popfits + 1000 * max(0, cons);
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate random indices for direction vectors
    idx = 1:NP;
    R = zeros(NP, 6);
    for i = 1:NP
        available = idx(idx ~= i & idx ~= elite_idx);
        R(i,:) = available(randperm(length(available), 6));
    end
    
    % Compute direction vectors
    d_local = x_elite(ones(NP,1), :) - popdecs;
    d_global = popdecs(R(:,1),:) - popdecs(R(:,2),:) + ...
               popdecs(R(:,3),:) - popdecs(R(:,4),:);
    d_constraint = tanh(abs(cons)) .* (popdecs(R(:,5),:) - popdecs(R(:,6),:));
    
    % 3. Compute adaptive weights based on fitness ranks
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    w1 = 0.5 + 0.3 * (ranks/NP);
    w2 = 0.3 - 0.2 * (ranks/NP);
    w3 = 0.2 + 0.1 * (ranks/NP);
    
    % Normalize weights
    w_sum = w1 + w2 + w3;
    w1 = w1 ./ w_sum;
    w2 = w2 ./ w_sum;
    w3 = w3 ./ w_sum;
    
    % 4. Compute adaptive scaling factors
    F = 0.5 + 0.3 * tanh(abs(cons)) .* randn(NP, 1);
    F = F(:, ones(1, D));
    
    % 5. Mutation with weighted directions
    mutants = popdecs + F .* (w1.*d_local + w2.*d_global + w3.*d_constraint);
    
    % 6. Rank-based adaptive crossover
    CR = 0.9 - 0.5 * (ranks/NP);
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
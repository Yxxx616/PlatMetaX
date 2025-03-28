% MATLAB Code
function [offspring] = updateFunc1240(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering both fitness and constraints
    penalty = popfits + 1000 * max(0, cons);
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate fitness-weighted direction vectors
    sigma_f = std(popfits) + eps;
    idx = 1:NP;
    R = zeros(NP, 4);
    for i = 1:NP
        available = idx(~ismember(idx, [i, elite_idx]));
        R(i,:) = available(randperm(length(available), 4));
    end
    
    % Compute weights based on fitness
    f_pairs = (popfits(R(:,1:2)) + popfits(R(:,3:4)))/2;
    w = exp(-f_pairs/sigma_f);
    w = w ./ (sum(w,2) + eps);
    
    % Compute weighted differences
    diff_vec = w(:,1).*(popdecs(R(:,1),:) - popdecs(R(:,2),:)) + ...
               w(:,2).*(popdecs(R(:,3),:) - popdecs(R(:,4),:));
    
    % 3. Constraint-aware scaling
    beta = 0.2 * tanh(abs(cons));
    F = 0.5 + beta .* randn(NP, 1);
    F = F(:, ones(1, D));
    
    % 4. Mutation with elite guidance
    elite_dir = (x_elite(ones(NP,1), :) - popdecs) / 2;
    epsilon = 0.1 * randn(NP, D);
    mutants = popdecs + F .* (elite_dir + diff_vec) + beta .* epsilon;
    
    % 5. Rank-based adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 - 0.4 * (ranks/NP);
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
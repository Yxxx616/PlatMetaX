% MATLAB Code
function [offspring] = updateFunc1300(popdecs, popfits, cons)
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
    
    % 2. Compute direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    % Tournament selection for random pairs (k=3)
    r1 = zeros(NP,1); r2 = zeros(NP,1);
    for i = 1:NP
        candidates = randperm(NP, 3);
        [~, idx] = min(popfits(candidates));
        r1(i) = candidates(idx);
        candidates = randperm(NP, 3);
        [~, idx] = min(popfits(candidates));
        r2(i) = candidates(idx);
    end
    d_random = popdecs(r1,:) - popdecs(r2,:);
    
    % 3. Adaptive weights combining fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    min_cons = min(cons);
    max_cons = max(cons);
    
    sigma = 0.7 * feasible + 0.3 * ~feasible;
    if max_fit > min_fit
        w_f = (popfits - min_fit) / (max_fit - min_fit);
    else
        w_f = zeros(NP,1);
    end
    
    if max_cons > min_cons
        w_c = (cons - min_cons) / (max_cons - min_cons);
    else
        w_c = zeros(NP,1);
    end
    
    weights = sigma .* w_f + (1-sigma) .* w_c;
    weights = weights / max(weights); % Normalize
    
    % 4. Mutation with adaptive F
    F = 0.7 + 0.1 * randn(NP,1);
    F = min(max(F, 0.5), 0.9);
    weights = weights(:, ones(1,D));
    F = F(:, ones(1,D));
    mutants = popdecs + F .* (weights.*d_elite + (1-weights).*d_random);
    
    % 5. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.5*(rank/NP);
    CR = CR(:, ones(1,D));
    
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
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1302(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution considering both fitness and constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, min_cons] = min(cons);
        [~, min_fit] = min(popfits);
        x_elite = (popdecs(min_cons,:) + popdecs(min_fit,:))/2;
    end
    
    % 2. Compute direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    % Fitness-based tournament selection (k=2)
    r1 = zeros(NP,1); r2 = zeros(NP,1);
    for i = 1:NP
        candidates = randperm(NP, 2);
        [~, idx] = min(popfits(candidates));
        r1(i) = candidates(idx);
        candidates = randperm(NP, 2);
        [~, idx] = min(popfits(candidates));
        r2(i) = candidates(idx);
    end
    d_fit = popdecs(r1,:) - popdecs(r2,:);
    
    % Constraint-based tournament selection (k=2)
    c1 = zeros(NP,1); c2 = zeros(NP,1);
    for i = 1:NP
        candidates = randperm(NP, 2);
        [~, idx] = min(cons(candidates));
        c1(i) = candidates(idx);
        candidates = randperm(NP, 2);
        [~, idx] = min(cons(candidates));
        c2(i) = candidates(idx);
    end
    d_cons = popdecs(c1,:) - popdecs(c2,:);
    
    % 3. Adaptive weights calculation
    min_fit = min(popfits);
    max_fit = max(popfits);
    min_cons = min(cons);
    max_cons = max(cons);
    
    w_fit = (popfits - min_fit) / (max_fit - min_fit + eps);
    w_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Normalize weights
    total = w_fit + w_cons + eps;
    alpha = w_fit ./ total;
    beta = w_cons ./ total;
    
    % 4. Adaptive scaling factor
    F = 0.5 + 0.3 * rand(NP,1) .* (1 - (alpha + beta)/2);
    F = min(max(F, 0.3), 0.9);
    
    % Expand dimensions for vectorized operations
    alpha = alpha(:, ones(1,D));
    beta = beta(:, ones(1,D));
    F = F(:, ones(1,D));
    
    % Combined mutation
    mutants = popdecs + F .* (alpha.*d_fit + beta.*d_cons + (1-alpha-beta).*d_elite);
    
    % 5. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.4*(rank/NP);
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
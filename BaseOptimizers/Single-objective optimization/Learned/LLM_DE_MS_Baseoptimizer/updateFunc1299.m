% MATLAB Code
function [offspring] = updateFunc1299(popdecs, popfits, cons)
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
    
    % Random pairs for differential vectors
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod((1:NP)+floor(NP/2)-1, NP)+1);
    d_random = popdecs(r1,:) - popdecs(r2,:);
    
    % 3. Adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    if max_fit > min_fit
        w_f = 0.4 + 0.4 * (popfits - min_fit) / (max_fit - min_fit);
    else
        w_f = 0.6 * ones(NP, 1);
    end
    
    min_cons = min(cons);
    max_cons = max(cons);
    if max_cons > min_cons
        w_c = 0.6 - 0.4 * (cons - min_cons) / (max_cons - min_cons);
    else
        w_c = 0.4 * ones(NP, 1);
    end
    
    weights = (w_f + w_c)/2;
    
    % 4. Dynamic scaling factor
    min_w = min(weights);
    max_w = max(weights);
    if max_w > min_w
        F = 0.5 + 0.3 * (weights - min_w) / (max_w - min_w);
    else
        F = 0.65 * ones(NP, 1);
    end
    
    % Mutation with adaptive weights and scaling
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
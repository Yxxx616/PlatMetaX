% MATLAB Code
function [offspring] = updateFunc99(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % 1. Weight calculation
    c_max = max(abs(cons));
    norm_cons = abs(cons) / (c_max + eps);
    
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = (1:NP)';
    norm_ranks = ranks / NP;
    
    alpha = 0.6;
    weights = alpha * norm_cons + (1-alpha) * (1 - norm_ranks);
    
    % 2. Elite selection
    [~, elite_idx] = sort(popfits + norm_cons);
    elites = popdecs(elite_idx(1:3), :);
    
    % 3. Mutation
    F = 0.8;
    v = zeros(NP, D);
    
    % Weighted tournament selection for r1, r2
    r1 = zeros(NP,1); r2 = r1;
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        prob = weights(candidates) / sum(weights(candidates));
        r1(i) = candidates(randsample(length(candidates), 1, true, prob));
        candidates = setdiff(candidates, r1(i));
        prob = weights(candidates) / sum(weights(candidates));
        r2(i) = candidates(randsample(length(candidates), 1, true, prob));
    end
    
    x_r1 = popdecs(r1, :);
    x_r2 = popdecs(r2, :);
    
    % Mutation vector
    v = elites(1,:) + F*(elites(2,:) - elites(3,:)) + ...
        F*(x_r1 - x_r2);
    
    % 4. Adaptive CR
    CR = 0.5 + 0.4 * (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    % 5. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
end
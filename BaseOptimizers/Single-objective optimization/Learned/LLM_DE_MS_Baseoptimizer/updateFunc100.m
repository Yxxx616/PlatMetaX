% MATLAB Code
function [offspring] = updateFunc100(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % 1. Weight calculation
    c_max = max(abs(cons));
    norm_cons = abs(cons) / (c_max + eps);
    
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = (1:NP)';
    norm_ranks = ranks / NP;
    
    alpha = 0.7;
    weights = alpha * norm_cons + (1-alpha) * (1 - norm_ranks);
    weights = weights / sum(weights); % Normalize
    
    % 2. Elite selection
    beta = 0.5;
    elite_scores = popfits + beta * abs(cons);
    [~, elite_idx] = sort(elite_scores);
    elites = popdecs(elite_idx(1:3), :);
    
    % 3. Mutation
    F = 0.8;
    v = zeros(NP, D);
    
    % Weighted selection for mutation vectors
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        [r1, r2, r3] = weightedSelection(candidates, weights, 3);
        [r1p, r2p, r3p] = weightedSelection(setdiff(candidates, [r1 r2 r3]), weights, 3);
        
        diff_vec = weights(r1)*(popdecs(r1,:)-popdecs(r1p,:)) + ...
                   weights(r2)*(popdecs(r2,:)-popdecs(r2p,:)) + ...
                   weights(r3)*(popdecs(r3,:)-popdecs(r3p,:));
        
        v(i,:) = elites(1,:) + F*(elites(2,:)-elites(3,:)) + F*diff_vec;
    end
    
    % 4. Adaptive CR
    CR = 0.4 + 0.5 * (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    % 5. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
end

function [selected] = weightedSelection(candidates, weights, n)
    selected = zeros(n,1);
    for k = 1:n
        cum_weights = cumsum(weights(candidates));
        r = rand() * cum_weights(end);
        idx = find(cum_weights >= r, 1);
        selected(k) = candidates(idx);
        candidates(idx) = [];
    end
end
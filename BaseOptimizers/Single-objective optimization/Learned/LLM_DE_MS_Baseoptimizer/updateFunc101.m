% MATLAB Code
function [offspring] = updateFunc101(popdecs, popfits, cons)
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
    weights = weights / sum(weights); % Normalize
    
    % 2. Elite selection (top 3 based on combined score)
    beta = 0.8;
    elite_scores = popfits + beta * abs(cons);
    [~, elite_idx] = sort(elite_scores);
    elites = popdecs(elite_idx(1:3), :);
    x_best = elites(1,:);
    
    % 3. Mutation
    F = 0.7;
    v = zeros(NP, D);
    
    for i = 1:NP
        % Select 3 distinct individuals (excluding current)
        candidates = setdiff(1:NP, i);
        idx = weightedSelection(candidates, weights(candidates), 3);
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Compute weighted direction vector
        w_sum = weights(r1) + weights(r2) + weights(r3);
        dir_vec = (weights(r1)*(popdecs(r1,:)-popdecs(i,:)) + ...
                  (weights(r2)*(popdecs(r2,:)-popdecs(i,:)) + ...
                  (weights(r3)*(popdecs(r3,:)-popdecs(i,:))) / w_sum;
        
        % Mutation vector
        v(i,:) = x_best + F*(elites(2,:)-elites(3,:)) + F*dir_vec;
    end
    
    % 4. Adaptive CR
    CR = 0.2 + 0.6 * (1 - norm_cons);
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
        cum_weights = cumsum(weights);
        r = rand() * cum_weights(end);
        idx = find(cum_weights >= r, 1);
        selected(k) = candidates(idx);
        candidates(idx) = [];
        weights(idx) = [];
    end
end
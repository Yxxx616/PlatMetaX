function [offspring] = updateFunc46(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.5;
    lambda = 0.1;
    top_ratio = 0.3;
    
    % Normalize constraint violations
    c_max = max(abs(cons));
    if c_max == 0
        c_max = 1; % avoid division by zero
    end
    c_weights = 1 - abs(cons)/c_max;
    
    % Sort by fitness to find best individuals
    [~, sorted_idx] = sort(popfits);
    best_idx = sorted_idx(1);
    top_group = sorted_idx(1:ceil(NP*top_ratio));
    
    for i = 1:NP
        % Select distinct vectors
        available = setdiff(1:NP, i);
        top_available = intersect(top_group, available);
        
        if isempty(top_available)
            r1 = randi(NP);
            while r1 == i
                r1 = randi(NP);
            end
        else
            r1 = top_available(randi(length(top_available)));
        end
        
        % Select r2 considering both fitness and constraints
        prob = abs(popfits(available)) + lambda*abs(cons(available));
        prob = prob / sum(prob);
        cum_prob = cumsum(prob);
        r = rand();
        r2_idx = find(cum_prob >= r, 1);
        r2 = available(r2_idx);
        
        % Mutation
        offspring(i,:) = popdecs(best_idx,:) + ...
                         F * (popdecs(r1,:) - popdecs(r2,:)) + ...
                         lambda * c_weights(i) * randn(1,D);
    end
end
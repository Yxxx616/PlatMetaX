function [offspring] = updateFunc48(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    lambda = 0.3;
    top_ratio = 0.3;
    diversity_weight = 0.2;
    
    % Normalize constraint violations
    c_abs = abs(cons);
    c_min = min(c_abs);
    c_max = max(c_abs);
    if c_max == c_min
        c_weights = zeros(NP,1);
    else
        c_weights = (c_abs - c_min) / (c_max - c_min);
    end
    
    % Sort by fitness to find best individuals and ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    best_idx = sorted_idx(1);
    top_group = sorted_idx(1:ceil(NP*top_ratio));
    other_group = setdiff(1:NP, top_group);
    
    for i = 1:NP
        % Select r1 from top performers
        available_top = setdiff(top_group, i);
        if isempty(available_top)
            r1 = randi(NP);
            while r1 == i
                r1 = randi(NP);
            end
        else
            r1 = available_top(randi(length(available_top)));
        end
        
        % Select r2 considering both fitness and constraints
        available = setdiff(1:NP, [i, r1]);
        if isempty(available)
            r2 = r1;
        else
            prob = abs(popfits(available)) + lambda*c_abs(available);
            prob = prob / sum(prob);
            cum_prob = cumsum(prob);
            r = rand();
            r2_idx = find(cum_prob >= r, 1);
            r2 = available(r2_idx);
        end
        
        % Select r3 from remaining population
        available_other = setdiff(other_group, [i, r1, r2]);
        if isempty(available_other)
            r3 = randi(NP);
            while r3 == i || r3 == r1 || r3 == r2
                r3 = randi(NP);
            end
        else
            r3 = available_other(randi(length(available_other)));
        end
        
        % Dynamic scaling factors
        F_base = 0.5 * (1 + c_weights(i));
        F_div = diversity_weight * (1 - ranks(i)/NP);
        
        % Mutation
        offspring(i,:) = popdecs(best_idx,:) + ...
                        F_base * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        F_div * (popdecs(r3,:) - popdecs(i,:));
    end
end
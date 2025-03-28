function [offspring] = updateFunc50(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    lambda = 0.5;
    top_ratio = 0.4;
    diversity_ratio = 0.2;
    
    % Normalize fitness values
    f_min = min(popfits);
    f_max = max(popfits);
    if f_max == f_min
        f_weights = zeros(NP,1);
    else
        f_weights = (popfits - f_min) / (f_max - f_min);
    end
    
    % Normalize constraint violations
    c_abs = abs(cons);
    c_min = min(c_abs);
    c_max = max(c_abs);
    if c_max == c_min
        c_weights = zeros(NP,1);
    else
        c_weights = (c_abs - c_min) / (c_max - c_min);
    end
    
    % Find best individual
    [~, best_idx] = min(popfits);
    
    % Create top performer group
    [~, sorted_idx] = sort(popfits);
    top_group = sorted_idx(1:ceil(NP*top_ratio));
    
    % Create diversity group (random selection)
    diversity_group = randperm(NP, ceil(NP*diversity_ratio));
    
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
        
        % Select r2 from top performers (different from r1)
        available_top = setdiff(top_group, [i, r1]);
        if isempty(available_top)
            r2 = randi(NP);
            while r2 == i || r2 == r1
                r2 = randi(NP);
            end
        else
            r2 = available_top(randi(length(available_top)));
        end
        
        % Select r3 considering both fitness and constraints
        available = setdiff(1:NP, [i, r1, r2]);
        if isempty(available)
            r3 = r1;
        else
            prob = abs(popfits(available)) + lambda*c_abs(available);
            prob = prob / sum(prob);
            cum_prob = cumsum(prob);
            r = rand();
            r3_idx = find(cum_prob >= r, 1);
            r3 = available(r3_idx);
        end
        
        % Select r4 randomly from diversity group
        available_div = setdiff(diversity_group, [i, r1, r2, r3]);
        if isempty(available_div)
            r4 = randi(NP);
            while r4 == i || r4 == r1 || r4 == r2 || r4 == r3
                r4 = randi(NP);
            end
        else
            r4 = available_div(randi(length(available_div)));
        end
        
        % Select r5 and r6 randomly
        available = setdiff(1:NP, [i, r1, r2, r3, r4]);
        if length(available) < 2
            r5 = randi(NP);
            r6 = randi(NP);
            while r5 == i || r5 == r1 || r5 == r2 || r5 == r3 || r5 == r4 || r5 == r6
                r5 = randi(NP);
            end
            while r6 == i || r6 == r1 || r6 == r2 || r6 == r3 || r6 == r4 || r6 == r5
                r6 = randi(NP);
            end
        else
            idxs = randperm(length(available), 2);
            r5 = available(idxs(1));
            r6 = available(idxs(2));
        end
        
        % Calculate scaling factors
        alpha = 0.5 * (1 + f_weights(i));
        beta = 0.3 * (1 + c_weights(i));
        gamma = 0.2 * (1 - f_weights(i)) * (1 - c_weights(i));
        
        % Mutation
        offspring(i,:) = popdecs(best_idx,:) + ...
                        alpha * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        beta * (popdecs(r3,:) - popdecs(r4,:)) + ...
                        gamma * (popdecs(r5,:) - popdecs(r6,:));
    end
end
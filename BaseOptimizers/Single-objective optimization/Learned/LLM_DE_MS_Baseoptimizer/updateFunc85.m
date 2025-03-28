function [offspring] = updateFunc85(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate combined weights considering both fitness and constraints
    weights = 1./(1 + abs(popfits) + max(0, cons));
    weights = weights / sum(weights);
    
    % Find best individual (max weight)
    [~, best_idx] = max(weights);
    
    % Calculate fitness ranks (lower rank = better fitness)
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    norm_ranks = (ranks - 1) / (NP - 1);
    
    % Normalize constraint violations
    max_cons = max(abs(cons));
    if max_cons == 0
        norm_cons = zeros(size(cons));
    else
        norm_cons = abs(cons) / max_cons;
    end
    
    for i = 1:NP
        % Tournament selection for base vector (3 candidates)
        candidates = setdiff(1:NP, i);
        tourn_size = min(3, length(candidates));
        tourn_idx = randperm(length(candidates), tourn_size);
        [~, best_tourn] = max(weights(candidates(tourn_idx)));
        base_idx = candidates(tourn_idx(best_tourn));
        
        % Select difference vectors using tournament selection
        candidates = setdiff(1:NP, [i, base_idx]);
        if length(candidates) < 2
            r = randsample(candidates, 2, true);
        else
            tourn_idx = randperm(length(candidates), min(4, length(candidates)));
            [~, sorted_tourn] = sort(weights(candidates(tourn_idx)), 'descend');
            selected = candidates(tourn_idx(sorted_tourn(1:2)));
            r = selected(randperm(2));
        end
        r1 = r(1); r2 = r(2);
        
        % Adaptive scaling factors
        F_i = 0.5 + 0.5 * (1 - norm_ranks(i));
        eta_i = 0.2 * (1 + norm_cons(i));
        
        % Generate mutation vector
        diff = popdecs(r1,:) - popdecs(r2,:);
        best_diff = popdecs(best_idx,:) - popdecs(i,:);
        
        mutant = popdecs(base_idx,:) + F_i * diff + eta_i * best_diff;
        
        offspring(i,:) = mutant;
    end
end
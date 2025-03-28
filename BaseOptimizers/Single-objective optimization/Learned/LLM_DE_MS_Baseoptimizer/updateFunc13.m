function [offspring] = updateFunc13(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -1000 * ones(1,D);
    ub = 1000 * ones(1,D);
    eps = 1e-10;
    
    % Normalize constraint violations
    cons_abs = abs(cons);
    cons_norm = (cons_abs - min(cons_abs)) / (max(cons_abs) - min(cons_abs) + eps);
    
    for i = 1:NP
        % Select five distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(NP-1, 5));
        r = idx(1:5);
        
        % Get fitness and constraint values
        f = popfits(r);
        c = cons_abs(r);
        
        % Compute weights
        f_sum = sum(abs(f)) + eps;
        w = abs(f) / f_sum;
        
        c_sum = sum(c) + eps;
        c_w = c / c_sum;
        
        % Create weighted base vector
        base = sum(popdecs(r,:) .* (w .* (1 - c_w)) / sum(w .* (1 - c_w) + eps);
        
        % Create difference vectors
        diff1 = zeros(1,D);
        diff2 = zeros(1,D);
        for j = 1:5
            diff1 = diff1 + w(j) * (popdecs(r(j),:) - popdecs(r(mod(j,5)+1,:));
            diff2 = diff2 + c_w(j) * (popdecs(r(j),:) - popdecs(r(mod(j+1,5)+1),:));
        end
        
        % Adaptive scaling factor
        F = 0.5 + 0.5*sin(pi*cons_norm(i));
        
        % Generate mutant vector
        mutant = base + F*(diff1 + diff2);
        
        % Bounce-back boundary handling
        for j = 1:D
            if mutant(j) < lb(j)
                mutant(j) = lb(j) + rand*(ub(j)-lb(j));
            elseif mutant(j) > ub(j)
                mutant(j) = ub(j) - rand*(ub(j)-lb(j));
            end
        end
        
        offspring(i,:) = mutant;
    end
end
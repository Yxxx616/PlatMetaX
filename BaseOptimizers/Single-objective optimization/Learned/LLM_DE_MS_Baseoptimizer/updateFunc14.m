function [offspring] = updateFunc14(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -1000 * ones(1,D);
    ub = 1000 * ones(1,D);
    eps = 1e-10;
    
    % Find best individual (considering constraints)
    [~, best_idx] = min(popfits + 1e6*(cons > 0));
    x_best = popdecs(best_idx,:);
    
    % Normalize constraint violations
    cons_abs = abs(cons);
    cons_norm = cons_abs / (max(cons_abs) + eps);
    
    for i = 1:NP
        % Select four distinct random indices
        candidates = 1:NP;
        candidates([i, best_idx]) = [];
        idx = candidates(randperm(length(candidates), 4));
        r = idx(1:4);
        
        % Get fitness and constraint values
        f = popfits(r);
        c = cons_abs(r);
        
        % Compute weights
        f_sum = sum(abs(f)) + eps;
        w_f = abs(f) / f_sum;
        
        c_sum = sum(c) + eps;
        w_c = c / c_sum;
        
        % Combined weights
        w = w_f .* (1 - w_c);
        w = w / (sum(w) + eps);
        
        % Create weighted base vector
        base = popdecs(r(1),:) + w(2)*(popdecs(r(2),:) - popdecs(r(3),:)) + ...
               w(3)*(popdecs(r(3),:) - popdecs(r(4),:));
        
        % Adaptive scaling factor
        F = 0.4 + 0.4 * cons_norm(i);
        
        % Generate mutant vector
        diff1 = x_best - popdecs(i,:);
        diff2 = popdecs(r(1),:) - popdecs(r(2),:);
        mutant = base + F*diff1 + 0.5*F*diff2;
        
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
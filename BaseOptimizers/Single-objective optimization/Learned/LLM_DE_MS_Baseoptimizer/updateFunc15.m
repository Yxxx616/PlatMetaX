function [offspring] = updateFunc15(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -1000 * ones(1,D);
    ub = 1000 * ones(1,D);
    eps = 1e-10;
    
    % Find best individual considering constraints
    [~, best_idx] = min(popfits + 1e6*(cons > 0));
    x_best = popdecs(best_idx,:);
    
    % Normalize constraint violations
    cons_abs = abs(cons);
    max_cons = max(cons_abs);
    cons_norm = cons_abs / (max_cons + eps);
    
    for i = 1:NP
        % Select four distinct random indices excluding current and best
        candidates = 1:NP;
        candidates([i, best_idx]) = [];
        idx = candidates(randperm(length(candidates), 4));
        r = idx(1:4);
        
        % Get fitness and constraint values
        f = popfits(r);
        c = cons_abs(r);
        
        % Compute fitness weights
        f_sum = sum(abs(f)) + eps;
        w_f = abs(f) / f_sum;
        
        % Compute constraint weights
        c_sum = sum(c) + eps;
        w_c = c / c_sum;
        
        % Combined weights
        w = w_f .* (1 - w_c);
        w = w / (sum(w) + eps);
        
        % Create weighted base vector
        base = popdecs(r(1),:) + w(2)*(popdecs(r(2),:) - popdecs(r(3),:)) + ...
               w(3)*(popdecs(r(3),:) - popdecs(r(4),:));
        
        % Adaptive scaling factor using tanh
        F = 0.5 + 0.3 * tanh(cons_norm(i));
        
        % Generate mutant vector
        diff1 = x_best - popdecs(i,:);
        diff2 = popdecs(r(1),:) - popdecs(r(2),:);
        mutant = base + F*diff1 + 0.5*F*diff2;
        
        % Boundary handling with bounce-back
        out_of_bounds = (mutant < lb) | (mutant > ub);
        rnd = rand(1,D);
        mutant(mutant < lb) = lb(mutant < lb) + rnd(mutant < lb).*(ub(mutant < lb)-lb(mutant < lb));
        mutant(mutant > ub) = ub(mutant > ub) - rnd(mutant > ub).*(ub(mutant > ub)-lb(mutant > ub));
        
        offspring(i,:) = mutant;
    end
end
function [offspring] = updateFunc71(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    F = 0.5;
    alpha = 0.3;
    
    % Find best constraint violation
    [~, best_cons_idx] = min(cons);
    c_best = cons(best_cons_idx);
    
    for i = 1:NP
        % Select four distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(NP-1, 4));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4);
        
        % Base vector is randomly selected
        base = popdecs(r1,:);
        
        % Differential component
        diff = popdecs(r2,:) - popdecs(r3,:);
        
        % Constraint-aware perturbation
        cons_term = alpha * sign(c_best - cons(i)) * popdecs(r4,:);
        
        % Combine components
        offspring(i,:) = base + F * diff + cons_term;
        
        % Fitness-based refinement
        if popfits(i) > median(popfits)
            offspring(i,:) = offspring(i,:) + 0.1*(popdecs(best_cons_idx,:) - popdecs(i,:));
        end
    end
    
    % Boundary control (optional, problem-dependent)
    % offspring = min(max(offspring, -1000), 1000);
end
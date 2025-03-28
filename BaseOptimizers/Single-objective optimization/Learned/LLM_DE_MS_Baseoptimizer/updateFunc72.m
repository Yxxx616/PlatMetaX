function [offspring] = updateFunc72(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    F = 0.5;
    alpha = 0.2;
    beta = 0.1;
    
    % Find best individual based on constraint violation
    [~, best_cons_idx] = min(cons);
    c_best = cons(best_cons_idx);
    
    % Find best individual based on fitness
    [~, best_fit_idx] = min(popfits);
    x_best = popdecs(best_fit_idx,:);
    
    % Sort population by fitness to get ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    w_rank = 1.0 + (ranks/NP - 0.5); % Weight between 0.5 and 1.5
    
    % Select top 30% individuals for base vectors
    top30 = floor(0.3*NP);
    if top30 < 1, top30 = 1; end
    top_idx = sorted_idx(1:top30);
    
    for i = 1:NP
        % Select base vector from top 30%
        base_idx = top_idx(randi(length(top_idx)));
        x_base = popdecs(base_idx,:);
        
        % Select two distinct random vectors
        candidates = 1:NP;
        candidates([i, base_idx]) = [];
        r = candidates(randperm(length(candidates), 2));
        x_r1 = popdecs(r(1),:);
        x_r2 = popdecs(r(2),:);
        
        % Select another random vector for constraint term
        r3 = candidates(randperm(length(candidates), 1));
        x_r3 = popdecs(r3,:);
        
        % Differential component with rank weight
        diff = (x_r1 - x_r2) * w_rank(i);
        
        % Constraint-aware perturbation
        cons_term = alpha * (c_best - cons(i)) * x_r3;
        
        % Fitness-driven refinement
        refine_term = beta * (x_best - popdecs(i,:));
        
        % Combine all components
        offspring(i,:) = x_base + F * diff + cons_term + refine_term;
    end
    
    % Boundary control
    offspring = min(max(offspring, -1000), 1000);
end
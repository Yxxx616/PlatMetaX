function [offspring] = updateFunc64(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate population statistics
    feasible = all(cons <= 0, 2);
    x_avg = mean(popdecs, 1);
    f_avg = mean(popfits);
    f_max = max(popfits);
    f_min = min(popfits);
    phi_avg = mean(cons);
    
    % Get best solution (considering feasibility)
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        feasible_idx = find(feasible);
        x_best = popdecs(feasible_idx(best_idx), :);
    else
        [~, best_idx] = min(popfits);
        x_best = popdecs(best_idx, :);
    end
    
    % Get worst solution
    [~, worst_idx] = max(popfits);
    x_worst = popdecs(worst_idx, :);
    
    % Pre-compute scaling factors
    F1 = 0.5 * (1 + (f_avg - popfits) ./ (f_max - f_min + eps));
    F2 = 0.8 * tanh(cons ./ (phi_avg + eps));
    
    % Generate random indices matrix
    r = zeros(NP, 4);
    for i = 1:NP
        r(i,:) = randperm(NP, 4);
    end
    
    % Vectorized mutation
    mask = cons <= phi_avg;
    offspring(mask,:) = x_best + F1(mask).*(popdecs(r(mask,1),:) - popdecs(r(mask,2),:)) + ...
                       F2(mask).*(popdecs(r(mask,3),:) - popdecs(r(mask,4),:));
    
    offspring(~mask,:) = x_avg + F1(~mask).*(popdecs(r(~mask,1),:) - popdecs(r(~mask,2),:)) + ...
                        F2(~mask).*(x_best - x_worst);
    
    % Boundary control with reflection
    lb = -500; ub = 500;
    out_of_bounds = offspring < lb | offspring > ub;
    offspring(out_of_bounds) = x_best(out_of_bounds) + ...
                              rand(sum(out_of_bounds(:)),1) .* ...
                              (min(ub,x_best(out_of_bounds)) - max(lb,x_best(out_of_bounds)));
end
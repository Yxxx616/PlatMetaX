function [offspring] = updateFunc65(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate feasibility and population statistics
    feasible = all(cons <= 0, 2);
    x_mean = mean(popdecs, 1);
    f_avg = mean(popfits);
    f_max = max(popfits);
    f_min = min(popfits);
    phi_avg = mean(abs(cons));
    
    % Get best solution (prioritizing feasible solutions)
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
    
    % Compute adaptive scaling factors
    F1 = 0.5 * (1 + (f_avg - popfits) ./ (f_max - f_min + eps));
    F2 = 0.8 * tanh(abs(cons) ./ (phi_avg + eps));
    
    % Generate random indices matrix (ensuring distinct indices)
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % Vectorized mutation based on constraint violation
    mask = abs(cons) <= phi_avg;
    
    % Mutation for individuals with small constraint violation
    offspring(mask,:) = x_best + F1(mask).*(popdecs(r(mask,1),:) - popdecs(r(mask,2),:)) + ...
                       F2(mask).*(popdecs(r(mask,3),:) - popdecs(r(mask,4),:));
    
    % Mutation for individuals with large constraint violation
    offspring(~mask,:) = x_mean + F1(~mask).*(popdecs(r(~mask,1),:) - popdecs(r(~mask,2),:)) + ...
                        F2(~mask).*(x_best - x_worst);
    
    % Boundary control with reflection and random reinitialization
    lb = -500; ub = 500;
    out_of_bounds = offspring < lb | offspring > ub;
    if any(out_of_bounds(:))
        % For out-of-bounds variables, reflect back or reinitialize randomly
        reflect_low = offspring < lb;
        reflect_high = offspring > ub;
        offspring(reflect_low) = 2*lb - offspring(reflect_low);
        offspring(reflect_high) = 2*ub - offspring(reflect_high);
        
        % If still out of bounds after reflection, reinitialize randomly
        still_out = offspring < lb | offspring > ub;
        offspring(still_out) = lb + (ub-lb)*rand(sum(still_out(:)),1);
    end
end
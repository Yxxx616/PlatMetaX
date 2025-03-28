% MATLAB Code
function [offspring] = updateFunc1596(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    cons_min = min(cons);
    cons_max = max(cons);
    
    F = 0.5 * (1 + (popfits - f_min) ./ (f_max - f_min + eps));
    C = (cons - cons_min) ./ (cons_max - cons_min + eps);
    CR = 0.9 - 0.5 * (popfits - f_min) ./ (f_max - f_min + eps);
    
    % 3. Generate random indices
    idx = arrayfun(@(x) randperm(NP, 4), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3); r4 = idx(:,4);
    
    % 4. Create mutation vectors
    mutation = elite + F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               C.*(popdecs(r3,:) - popdecs(r4,:));
    
    % 5. Boundary handling with adaptive bounce-back
    lb_mask = mutation < lb;
    ub_mask = mutation > ub;
    mutation(lb_mask) = (popdecs(lb_mask) + lb(lb_mask))/2 + ...
                        rand(sum(lb_mask(:)),1).*(popdecs(lb_mask) - lb(lb_mask))/2;
    mutation(ub_mask) = (popdecs(ub_mask) + ub(ub_mask))/2 - ...
                        rand(sum(ub_mask(:)),1).*(ub(ub_mask) - popdecs(ub_mask))/2;
    
    % 6. Crossover
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % 7. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
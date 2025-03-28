% MATLAB Code
function [offspring] = updateFunc1595(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite and feasible solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible(best_feas_idx), :);
        [~, worst_feas_idx] = max(popfits(feasible));
        worst_feas = popdecs(feasible(worst_feas_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
        best_feas = elite;
        worst_feas = popdecs(randi(NP), :);
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.4 + 0.4 * (popfits - f_min) ./ (f_max - f_min + eps);
    C = cons ./ (max(cons) + eps);
    
    % 3. Generate random indices
    idx1 = randi(NP, NP, 1);
    idx2 = randi(NP, NP, 1);
    same_idx = idx1 == idx2;
    idx2(same_idx) = mod(idx2(same_idx) + randi(NP-1, sum(same_idx), 1), NP);
    
    % 4. Create mutation vectors
    elite_terms = elite + F .* (popdecs(idx1,:) - popdecs(idx2,:));
    feas_terms = best_feas - worst_feas;
    mutation = elite_terms + C .* feas_terms;
    
    % 5. Boundary handling with bounce-back
    lb_mask = mutation < lb;
    ub_mask = mutation > ub;
    mutation(lb_mask) = lb(lb_mask) + rand(sum(lb_mask(:)),1) .* ...
                       (popdecs(lb_mask) - lb(lb_mask));
    mutation(ub_mask) = ub(ub_mask) - rand(sum(ub_mask(:)),1) .* ...
                       (ub(ub_mask) - popdecs(ub_mask));
    
    % 6. Adaptive crossover
    CR = 0.8 - 0.6 * (popfits - f_min) / (f_max - f_min + eps);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % 7. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
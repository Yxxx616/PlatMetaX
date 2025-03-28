% MATLAB Code
function [offspring] = updateFunc905(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select best individual (feasible first)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx,:);
    end
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r = candidates(randperm(length(candidates), 4));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3); r4(i) = r(4);
    end
    
    % 3. Compute adaptive parameters
    mean_fit = mean(popfits);
    F = 0.5 * (1 + tanh(popfits - mean_fit));
    
    % Constraint weights
    max_cons = max(abs(cons)) + eps;
    c_weights = tanh(abs(cons)/max_cons);
    
    % 4. Compute mutation components
    % Fitness-guided direction
    df = bsxfun(@minus, x_best, popdecs) + 0.1*(popdecs(r1,:) - popdecs(r2,:));
    
    % Constraint-aware direction
    dc = bsxfun(@times, c_weights, popdecs(r3,:) - popdecs(r4,:));
    
    % Diversity direction
    centroid = mean(popdecs, 1);
    dd = bsxfun(@minus, centroid, popdecs) .* (0.1 * rand(NP,D));
    
    % 5. Composite mutation
    mutants = popdecs + bsxfun(@times, F, df) + ...
              bsxfun(@times, (1-F), dc) + dd;
    
    % 6. Adaptive crossover
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.3 + 0.5 * (ranks/NP).^2;
    
    % 7. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc1699(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (positive values indicate violation)
    pos_cons = max(0, cons);
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    w_cons = (pos_cons - c_min) / (c_max - c_min + eps);
    
    % Normalize fitness (0 is best)
    f_min = min(popfits);
    f_max = max(popfits);
    w_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    feasible_mask = pos_cons == 0;
    if any(feasible_mask)
        feas_pool = find(feasible_mask);
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas_idx = feas_pool(best_feas_idx);
        base_vec = popdecs(best_feas_idx,:);
    else
        [~, best_feas_idx] = min(pos_cons);
        base_vec = popdecs(best_idx,:);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Adaptive parameters
    F1 = 0.8 * (1 - w_fit) .* (1 - 0.5 * w_cons);
    F2 = 0.4 * (w_fit + 0.5 * w_cons);
    CR = 0.5 + 0.3 * (1 - w_fit) .* (1 - w_cons);
    j_rand = randi(D, NP, 1);
    
    % Pre-generate random indices (4 per individual)
    idx_matrix = zeros(NP, 4);
    for i = 1:NP
        idx_pool = setdiff(1:NP, i);
        idx_matrix(i,:) = idx_pool(randperm(length(idx_pool), 4));
    end
    
    for i = 1:NP
        % Get random indices
        r = idx_matrix(i,:);
        
        % Enhanced mutation with two difference vectors
        mutation = base_vec + F1(i) * (popdecs(r(1),:) - popdecs(r(2),:)) + ...
                  F2(i) * (popdecs(r(3),:) - popdecs(r(4),:));
        
        % Directional perturbation based on constraints
        mutation = mutation + 0.1 * sign(cons(i)) .* (ub-lb) .* randn(1,D);
        
        % Binomial crossover
        mask = rand(1,D) <= CR(i) | (1:D) == j_rand(i);
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with reflection
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        offspring(i,viol_low) = 2*lb(viol_low) - offspring(i,viol_low);
        offspring(i,viol_high) = 2*ub(viol_high) - offspring(i,viol_high);
        
        % Ensure final bounds
        offspring(i,:) = min(max(offspring(i,:), lb), ub);
    end
    
    % Additional diversity preservation
    perturb_mask = rand(NP,D) < 0.05;
    offspring(perturb_mask) = lb(perturb_mask) + (ub(perturb_mask)-lb(perturb_mask)) .* rand(sum(perturb_mask(:)),1);
end
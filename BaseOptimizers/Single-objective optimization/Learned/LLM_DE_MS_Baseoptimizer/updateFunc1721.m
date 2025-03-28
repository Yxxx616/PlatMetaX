% MATLAB Code
function [offspring] = updateFunc1721(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (positive values indicate violation)
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    if any(feasible_mask)
        feas_pool = find(feasible_mask);
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas_idx = feas_pool(best_feas_idx);
        base_vec = popdecs(best_feas_idx,:);
    else
        [~, least_viol_idx] = min(pos_cons);
        base_vec = popdecs(least_viol_idx,:);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_avg = mean(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(5*(popfits - f_avg)/(f_max - f_min + eps));
    alpha = 0.4 * (1 - exp(-2 * norm_cons));
    CR = 0.9 - 0.5 * norm_cons;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    j_rand = randi(D, NP, 1);
    
    % Precompute all random indices with constraint-aware selection
    idx_matrix = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        con_probs = exp(-3*pos_cons(available));
        con_probs = con_probs / sum(con_probs);
        selected = randsample(available, 4, true, con_probs);
        idx_matrix(i,:) = selected;
    end
    
    % Vectorized mutation and crossover
    for i = 1:NP
        r = idx_matrix(i,:);
        mutation = base_vec + ...
                  F(i) * (popdecs(r(1),:) - popdecs(r(2),:)) + ...
                  alpha(i) * (popdecs(r(3),:) - popdecs(r(4),:));
        
        % Binomial crossover
        mask = rand(1,D) <= CR(i) | (1:D) == j_rand(i);
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with reflection
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        offspring(i,viol_low) = 2*lb(viol_low) - offspring(i,viol_low);
        offspring(i,viol_high) = 2*ub(viol_high) - offspring(i,viol_high);
    end
    
    % Additional diversity for highly constrained solutions
    high_constrained = norm_cons > 0.7;
    if any(high_constrained)
        perturb_mag = 0.15 * (ub - lb) .* randn(sum(high_constrained), D);
        offspring(high_constrained,:) = offspring(high_constrained,:) + perturb_mag;
        offspring = min(max(offspring, lb), ub);
    end
end
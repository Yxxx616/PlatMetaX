% MATLAB Code
function [offspring] = updateFunc1707(popdecs, popfits, cons)
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
        [~, best_feas_idx] = min(pos_cons);
        base_vec = popdecs(best_feas_idx,:);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Compute selection probabilities based on ranks
    [~, fit_rank_idx] = sort(popfits);
    fit_rank = zeros(NP,1);
    fit_rank(fit_rank_idx) = (1:NP)/NP;
    
    [~, cons_rank_idx] = sort(pos_cons);
    cons_rank = zeros(NP,1);
    cons_rank(cons_rank_idx) = (1:NP)/NP;
    
    % Adaptive parameters
    alpha = 0.7 * (1 - norm_cons);
    F = 0.5 + 0.3 * norm_fits;
    beta = 0.2 * norm_cons;
    CR = 0.9 - 0.4 * norm_fits;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    j_rand = randi(D, NP, 1);
    
    % Generate all random indices at once (vectorized)
    idx_matrix = zeros(NP, 4);
    for i = 1:NP
        % Fitness-based selection for r1, r2
        fit_probs = 1./(1 + fit_rank);
        fit_probs(i) = 0;
        fit_probs = fit_probs / sum(fit_probs);
        idx_matrix(i,1:2) = randsample(NP, 2, true, fit_probs);
        
        % Constraint-based selection for r3, r4
        cons_probs = 1./(1 + cons_rank);
        cons_probs(i) = 0;
        cons_probs = cons_probs / sum(cons_probs);
        idx_matrix(i,3:4) = randsample(NP, 2, true, cons_probs);
    end
    
    % Vectorized mutation and crossover
    for i = 1:NP
        r = idx_matrix(i,:);
        weighted_base = alpha(i) * popdecs(best_idx,:) + (1-alpha(i)) * base_vec;
        mutation = weighted_base + F(i) * (popdecs(r(1),:) - popdecs(r(2),:)) + ...
                  beta(i) * (popdecs(r(3),:) - popdecs(r(4),:));
        
        % Binomial crossover
        mask = rand(1,D) <= CR(i) | (1:D) == j_rand(i);
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with bounce-back
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        offspring(i,viol_low) = lb(viol_low) + rand(1,sum(viol_low)) .* ...
                               (popdecs(i,viol_low) - lb(viol_low));
        offspring(i,viol_high) = ub(viol_high) - rand(1,sum(viol_high)) .* ...
                                (ub(viol_high) - popdecs(i,viol_high));
    end
    
    % Small random perturbation for diversity
    perturb_mask = (norm_fits + norm_cons) > 1.5;
    if any(perturb_mask)
        perturb_amount = 0.1 * (rand(sum(perturb_mask),D) - 0.5) .* (ub - lb);
        offspring(perturb_mask,:) = offspring(perturb_mask,:) + perturb_amount;
        offspring = min(max(offspring, lb), ub);
    end
end
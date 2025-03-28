% MATLAB Code
function [offspring] = updateFunc1718(popdecs, popfits, cons)
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
    f_avg = mean(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Adaptive parameters
    alpha = 1 ./ (1 + exp(-5*(norm_cons - 0.5)));  % Sigmoid transition
    F_f = 0.5 + 0.3 * tanh(2*(popfits - f_avg)/(f_max - f_min + eps));
    F_c = 0.3 * (1 - exp(-2 * norm_cons));
    CR = 0.9 - 0.5 * norm_fits;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    j_rand = randi(D, NP, 1);
    
    % Precompute all random indices (vectorized)
    idx_matrix = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        
        % Fitness-based selection (better solutions more likely)
        fit_probs = exp(-4*(1:length(available))/sum(exp(-4*(1:length(available))));
        idx_matrix(i,1:2) = available(randsample(length(available), 2, true, fit_probs));
        
        % Constraint-based selection (less constrained more likely)
        con_probs = exp(-4*(1:length(available))/sum(exp(-4*(1:length(available))));
        idx_matrix(i,3:4) = available(randsample(length(available), 2, true, con_probs));
    end
    
    % Vectorized mutation and crossover
    for i = 1:NP
        r = idx_matrix(i,:);
        weighted_base = alpha(i) * popdecs(best_idx,:) + (1-alpha(i)) * base_vec;
        mutation = weighted_base + ...
                  F_f(i) * (popdecs(r(1),:) - popdecs(r(2),:)) + ...
                  F_c(i) * (popdecs(r(3),:) - popdecs(r(4),:));
        
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
        perturb_mag = 0.2 * (ub - lb) .* rand(sum(high_constrained), D);
        offspring(high_constrained,:) = offspring(high_constrained,:) + ...
            (rand(sum(high_constrained), D) - 0.5) .* perturb_mag;
        offspring = min(max(offspring, lb), ub);
    end
end
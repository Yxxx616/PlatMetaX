% MATLAB Code
function [offspring] = updateFunc1722(popdecs, popfits, cons)
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
    F = 0.5 + 0.2 * tanh(10*(popfits - f_avg)/(f_max - f_min + eps));
    alpha = 0.3 * (1 - exp(-5 * norm_cons));
    CR = 0.9 - 0.6 * norm_cons;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    j_rand = randi(D, NP, 1);
    
    % Precompute all random indices with constraint-aware selection
    idx_matrix = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        if any(feasible_mask(available))
            feas_available = available(feasible_mask(available));
            r1r2 = randsample(feas_available, 2);
            r3r4 = randsample(available, 2);
        else
            weights = exp(-5*pos_cons(available));
            weights = weights / sum(weights);
            r1r2 = randsample(available, 2, true, weights);
            r3r4 = randsample(available, 2);
        end
        idx_matrix(i,:) = [r1r2(1), r1r2(2), r3r4(1), r3r4(2)];
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
        
        % Boundary handling with random reinitialization for extreme violations
        viol = offspring(i,:) < lb | offspring(i,:) > ub;
        if norm_cons(i) > 0.8 && any(viol)
            offspring(i,viol) = lb(viol) + rand(1,sum(viol)) .* (ub(viol)-lb(viol));
        else
            offspring(i,offspring(i,:) < lb) = lb(offspring(i,:) < lb);
            offspring(i,offspring(i,:) > ub) = ub(offspring(i,:) > ub);
        end
    end
    
    % Additional perturbation for highly constrained solutions
    high_constrained = norm_cons > 0.7;
    if any(high_constrained)
        perturb_mag = 0.2 * (ub - lb) .* (rand(sum(high_constrained), D) - 0.5);
        offspring(high_constrained,:) = offspring(high_constrained,:) + perturb_mag;
        offspring = min(max(offspring, lb), ub);
    end
end
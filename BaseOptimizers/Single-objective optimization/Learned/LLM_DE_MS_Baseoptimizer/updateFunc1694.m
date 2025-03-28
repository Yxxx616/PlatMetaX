% MATLAB Code
function [offspring] = updateFunc1694(popdecs, popfits, cons)
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
    
    % Adaptive alpha based on feasibility ratio
    feas_ratio = sum(pos_cons == 0) / NP;
    alpha = 0.3 + 0.5 * (1 - feas_ratio); % More focus on constraints when few feasible
    
    % Combined weights
    weights = alpha*w_cons + (1-alpha)*w_fit;
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    feasible_mask = pos_cons == 0;
    if any(feasible_mask)
        feas_pool = find(feasible_mask);
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas_idx = feas_pool(best_feas_idx);
    else
        [~, best_feas_idx] = min(pos_cons);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-generate random numbers for vectorization
    F = 0.5 + 0.2*randn(NP,1);
    beta = 0.2 + 0.5*rand(NP,1);
    CR = 0.9*(1 - weights) + 0.1;
    j_rand = randi(D, NP, 1);
    
    for i = 1:NP
        % Select distinct random vectors (excluding current)
        idx_pool = setdiff(1:NP, i);
        r = idx_pool(randperm(length(idx_pool), 4));
        
        % Base vector construction
        x_base = popdecs(best_idx,:) + beta(i)*(1-weights(i))*(popdecs(r(1),:) - popdecs(r(2),:));
        
        % Direction vectors
        d_best = popdecs(best_idx,:) - popdecs(i,:);
        d_feas = popdecs(best_feas_idx,:) - popdecs(i,:);
        
        % Adaptive mutation with dynamic coefficients
        gamma1 = 0.5 + 0.3*rand();
        gamma2 = 0.3 + 0.4*rand();
        mutation = x_base + F(i)*(popdecs(r(3),:) - popdecs(r(4),:)) + ...
                  gamma1*weights(i)*d_best + gamma2*(1-w_cons(i))*d_feas;
        
        % Binomial crossover
        mask = rand(1,D) <= CR(i) | (1:D) == j_rand(i);
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with adaptive reflection
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        eta = 0.3 + 0.4*rand(1,D);
        offspring(i,viol_low) = lb(viol_low) + eta(viol_low).*(popdecs(i,viol_low) - lb(viol_low));
        offspring(i,viol_high) = ub(viol_high) - eta(viol_high).*(ub(viol_high) - popdecs(i,viol_high));
    end
    
    % Final adaptive perturbation for constrained solutions
    perturb_mask = rand(NP,D) < repmat(0.1 + 0.2*w_cons,1,D);
    offspring(perturb_mask) = offspring(perturb_mask) + ...
                            0.05*(ub(perturb_mask)-lb(perturb_mask)).*randn(sum(perturb_mask(:)),1);
    offspring = min(max(offspring, lb), ub);
end
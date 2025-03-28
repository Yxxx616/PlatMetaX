% MATLAB Code
function [offspring] = updateFunc173(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute constraint direction vector
    c_weights = max(0, abs(cons)); % Absolute constraint violations
    c_sum = sum(c_weights);
    if c_sum > eps
        d_c = (c_weights' * popdecs) / c_sum;
    else
        d_c = mean(popdecs, 1);
    end
    
    % 2. Compute fitness-weighted centroid
    min_fit = min(popfits);
    w_f = 1 ./ (1 + popfits - min_fit);
    x_w = (w_f' * popdecs) / sum(w_f);
    
    % 3. Generate random indices (vectorized)
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2);
    
    % 4. Compute adaptive parameters
    alpha = 0.5 * (1 - exp(-abs(cons))); % Constraint adaptation
    beta = 0.4 * c_weights; % Constraint weight
    gamma = 0.4 * w_f; % Fitness weight
    F = 0.6; % Base scaling factor
    
    % 5. Vectorized mutation
    for i = 1:NP
        base = popdecs(i,:);
        diff = popdecs(r1(i),:) - popdecs(r2(i),:);
        cons_term = beta(i) * (d_c - base);
        fit_term = gamma(i) * (x_w - base);
        
        % Enhanced differential term with constraint adaptation
        diff = diff .* (1 + alpha(i) * sign(cons(i)));
        
        v = base + F * diff + cons_term + fit_term;
        
        % 6. Adaptive crossover
        CR = 0.9 - 0.4 * (w_f(i) - min(w_f)) / (max(w_f) - min(w_f) + eps);
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = base;
        offspring(i,mask) = v(mask);
    end
    
    % 7. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    alpha_ref = 0.2 + 0.6 * rand(NP, D);
    offspring(below_lb) = lb_rep(below_lb) + alpha_ref(below_lb) .* ...
                        (offspring(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - alpha_ref(above_ub) .* ...
                        (offspring(above_ub) - ub_rep(above_ub));
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
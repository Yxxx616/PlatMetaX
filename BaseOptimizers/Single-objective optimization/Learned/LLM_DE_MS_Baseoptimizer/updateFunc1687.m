% MATLAB Code
function [offspring] = updateFunc1687(popdecs, popfits, cons)
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
    
    % Combined ranking (fitness + constraints)
    combined = w_fit + w_cons;
    [~, rank_idx] = sort(combined);
    elite_size = max(3, round(NP*0.2));
    elite_pool = rank_idx(1:elite_size);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-generate random numbers for vectorization
    rand_vals = rand(NP, D+2);
    rand_idx = randi(NP, NP, 4);
    
    for i = 1:NP
        % Select base vector from elite pool
        base_idx = elite_pool(randi(length(elite_pool)));
        x_base = popdecs(base_idx, :);
        
        % Select distinct random vectors
        idx_pool = setdiff(1:NP, [i, base_idx]);
        r = idx_pool(randperm(length(idx_pool), 2));
        r1 = r(1); r2 = r(2);
        
        % Adaptive parameters
        F = 0.5 + 0.3 * randn();
        sigma = 0.5 * (1 - w_fit(i));
        
        % Enhanced mutation with constraint and fitness awareness
        mutation = x_base + F * (popdecs(r1,:) - popdecs(r2,:)) .* (1 - w_cons(i)) + ...
                  sigma * (popdecs(rank_idx(1),:) - popdecs(i,:)) .* w_fit(i) + ...
                  0.1 * randn(1,D) .* (ub - lb) .* w_cons(i);
        
        % Adaptive exponential crossover
        j_rand = randi(D);
        L = 1 + sum(cumprod(rand_vals(i,1:D)) < (0.2 + 0.5*(1-w_cons(i))*(1-w_fit(i)));
        mask = false(1,D);
        mask(mod(j_rand:j_rand+L-1, D)+1) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Smart boundary handling
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        if any(viol_low | viol_high)
            offspring(i,viol_low) = 0.5*(lb(viol_low) + popdecs(i,viol_low));
            offspring(i,viol_high) = 0.5*(ub(viol_high) + popdecs(i,viol_high));
        end
    end
    
    % Final bounds enforcement with small perturbation
    out_of_bounds = offspring < lb | offspring > ub;
    offspring(out_of_bounds) = lb(out_of_bounds) + rand(1,sum(out_of_bounds(:))) .* ...
                              (ub(out_of_bounds) - lb(out_of_bounds));
end
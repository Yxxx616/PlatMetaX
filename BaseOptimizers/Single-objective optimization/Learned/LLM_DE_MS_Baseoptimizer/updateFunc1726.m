% MATLAB Code
function [offspring] = updateFunc1726(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Select base vector
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        base_vec = popdecs(feas_idx(best_idx),:);
    else
        [~, best_idx] = min(pos_cons);
        base_vec = popdecs(best_idx,:);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Compute population mean and directions
    pop_mean = mean(popdecs, 1);
    d_mean = popdecs - pop_mean(ones(NP,1),:);
    d_best = base_vec - pop_mean;
    
    % Generate random pairs
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive parameters
    F = 0.4 + 0.4 * tanh(5 * norm_fits);
    alpha = 0.3 * (1 - exp(-3 * norm_cons));
    CR = 0.9 - 0.4 * norm_cons;
    
    % Mutation
    mutation = base_vec(ones(NP,1),:) + ...
               bsxfun(@times, F, d_best(ones(NP,1),:) + d_mean) + ...
               bsxfun(@times, alpha .* randn(NP,1), d_rand);
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    offspring = min(max(offspring, lb_matrix), ub_matrix);
    
    % Elite preservation for best feasible
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        offspring(feas_idx(best_idx),:) = popdecs(feas_idx(best_idx),:);
    end
end
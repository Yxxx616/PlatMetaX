% MATLAB Code
function [offspring] = updateFunc1728(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Select base vector (best feasible or least infeasible)
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
    
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    norm_cons = (pos_cons - c_min) / (c_max - c_min + eps);
    
    % Compute population mean
    pop_mean = mean(popdecs, 1);
    
    % Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    while any(r3 == r4 | r3 == r1 | r3 == r2 | r4 == r1 | r4 == r2)
        r3 = randi(NP, NP, 1);
        r4 = randi(NP, NP, 1);
    end
    
    % Adaptive parameters
    F1 = 0.6 * (1 - tanh(5 * norm_fits));
    F2 = 0.4 * norm_cons;
    F3 = 0.2 * randn(NP, 1);
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation
    mutation = base_vec(ones(NP,1),:) + ...
               bsxfun(@times, F1, base_vec - pop_mean) + ...
               bsxfun(@times, F2, popdecs(r1,:) - popdecs(r2,:)) + ...
               bsxfun(@times, F3, popdecs(r3,:) - popdecs(r4,:));
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling
    offspring = min(max(offspring, lb), ub);
    
    % Elite preservation for best feasible
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        offspring(feas_idx(best_idx),:) = popdecs(feas_idx(best_idx),:);
    end
end
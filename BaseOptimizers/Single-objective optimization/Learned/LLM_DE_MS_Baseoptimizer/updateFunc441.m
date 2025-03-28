% MATLAB Code
function [offspring] = updateFunc441(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Identify elite solution
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(norm_fits + norm_cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate feasible center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
    end
    
    % Generate random indices without repetition
    idx = 1:NP;
    r1 = zeros(NP,1);
    r2 = zeros(NP,1);
    for i = 1:NP
        available = idx(idx ~= i);
        r1(i) = available(randi(NP-1));
        available = available(available ~= r1(i));
        r2(i) = available(randi(NP-2));
    end
    
    % Adaptive weights
    w1 = 0.5 * (1 - rho) * (1 - norm_cons);
    w2 = 0.7 * rho * (1 - norm_fits);
    w3 = 0.3 + 0.4 * rand(NP, 1);
    
    % Expand weights to D dimensions
    w1 = repmat(w1, 1, D);
    w2 = repmat(w2, 1, D);
    w3 = repmat(w3, 1, D);
    
    % Direction vectors
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    elite_dir = repmat(elite, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Combined mutation
    offspring = popdecs + w1.*feas_dir + w2.*elite_dir + w3.*div_dir;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for out-of-bound solutions
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low + ...
        (2*ub_rep - offspring) .* mask_high;
    
    % Final clipping to ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
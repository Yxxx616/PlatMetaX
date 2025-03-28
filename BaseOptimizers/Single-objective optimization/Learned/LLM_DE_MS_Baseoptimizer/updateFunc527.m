% MATLAB Code
function [offspring] = updateFunc527(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = abs_cons / (c_max + eps);
    
    % Calculate adaptive weights
    w = (1 - norm_fit).^2 .* exp(-norm_con);
    
    % Select elite individual
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Generate distinct random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) idx(randperm(NP-1,1)+(idx(randperm(NP-1,1))>=i), idx);
    r2 = arrayfun(@(i) idx(randperm(NP-1,1)+(idx(randperm(NP-1,1))>=i)), idx);
    r3 = arrayfun(@(i) idx(randperm(NP-1,1)+(idx(randperm(NP-1,1))>=i), idx);
    r4 = arrayfun(@(i) idx(randperm(NP-1,1)+(idx(randperm(NP-1,1))>=i), idx);
    
    % Calculate scaling factors
    F1 = 0.8 * w;
    F2 = 0.4 * (1 - w);
    F3 = 0.2 * (1 - w).^2;
    
    % Mutation operation
    diff1 = repmat(elite, NP, 1) - popdecs;
    diff2 = popdecs(r1,:) - popdecs(r2,:);
    diff3 = popdecs(r3,:) - popdecs(r4,:);
    
    offspring = popdecs + repmat(F1, 1, D) .* diff1 + ...
                repmat(F2, 1, D) .* diff2 + ...
                repmat(F3, 1, D) .* diff3;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    % For values below bound
    reflected_low = lb_rep + (lb_rep - offspring);
    offspring = offspring .* ~below + reflected_low .* below;
    
    % For values above bound
    reflected_high = ub_rep - (offspring - ub_rep);
    offspring = offspring .* ~above + reflected_high .* above;
    
    % Final projection
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Small random perturbation near boundaries
    boundary_threshold = 0.1 * (ub_rep - lb_rep);
    near_bound = (offspring - lb_rep < boundary_threshold) | ...
                (ub_rep - offspring < boundary_threshold);
    offspring = offspring + near_bound .* (rand(NP,D)-0.5).*boundary_threshold;
    offspring = max(min(offspring, ub_rep), lb_rep);
end
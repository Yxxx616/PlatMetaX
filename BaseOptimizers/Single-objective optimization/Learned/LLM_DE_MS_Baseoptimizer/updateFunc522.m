% MATLAB Code
function [offspring] = updateFunc522(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalized fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = abs_cons / (c_max + eps);
    
    % Weight calculation
    w = (1 - norm_fit).^2 ./ (1 + norm_con);
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_fit + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Index selection
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = mod(idx(1:NP) + floor(NP/2), NP) + 1;
    
    % Adaptive scaling factors
    F_elite = 0.7 * w.^1.2;
    F_rand = 0.5 * (1 - w).^0.9;
    sigma = 0.15 * (ub(1)-lb(1)) * (1 - w);
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Mutation
    offspring = popdecs + repmat(F_elite, 1, D) .* elite_dir + ...
                repmat(F_rand, 1, D) .* rand_dir + ...
                repmat(sigma, 1, D) .* randn(NP, D);
    
    % Boundary handling with memory
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflected = (lb_rep + repmat(w, 1, D) .* (popdecs - lb_rep)) .* below + ...
                (ub_rep - repmat(w, 1, D) .* (ub_rep - popdecs)) .* above;
    
    offspring = offspring .* (~below & ~above) + reflected;
    
    % Diversity enhancement
    rand_mask = rand(NP, D) < 0.02 * repmat(1-w, 1, D);
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring = offspring .* ~rand_mask + rand_vals .* rand_mask;
    
    % Final projection
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc516(popdecs, popfits, cons)
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
    
    % Calculate combined weights
    w = (1 - norm_fit).^1.5 .* (1 + norm_con).^0.7;
    
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
    
    % Random indices
    idx = randperm(NP);
    r1 = mod(idx, NP) + 1;
    r2 = mod(idx+1, NP) + 1;
    
    % Mutation parameters
    F1 = 0.7 * w;
    F2 = 0.5 * (1 - w);
    F3 = 0.3 * sqrt(w);
    sigma = 0.2 * (ub(1) - lb(1)) * (1 - w);
    
    % Vectorized mutation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    pert_dir = repmat(sigma, 1, D) .* randn(NP, D);
    
    offspring = popdecs + repmat(F1, 1, D) .* elite_dir + ...
                repmat(F2, 1, D) .* diff_dir + ...
                repmat(F3, 1, D) .* pert_dir;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    rand_factor = repmat(w, 1, D) .* rand(NP, D);
    reflected = (lb_rep + rand_factor .* (ub_rep - lb_rep)) .* below + ...
                (ub_rep - rand_factor .* (ub_rep - lb_rep)) .* above;
    
    offspring = offspring .* (~below & ~above) + reflected;
    offspring = max(min(offspring, ub_rep), lb_rep);
end
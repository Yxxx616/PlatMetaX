% MATLAB Code
function [offspring] = updateFunc521(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Enhanced weight calculation
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = log1p(abs_cons) / (log1p(c_max) + eps);
    w = (1 - norm_fit).^2.0 ./ (1 + norm_con);
    
    % Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_fit + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Index selection with anti-cycling
    [~, idx] = sort(rand(NP,1));
    r1 = mod(idx, NP) + 1;
    r2 = mod(idx + floor(NP/2), NP) + 1;
    
    % Adaptive parameters
    F1 = 0.8 * w.^1.5;
    F2 = 0.6 * (1 - w).^0.8;
    sigma = 0.1 * (ub(1)-lb(1)) * (1 - w);
    
    % Vectorized mutation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    pert_dir = repmat(sigma, 1, D) .* randn(NP, D);
    
    offspring = popdecs + repmat(F1, 1, D) .* elite_dir + ...
                repmat(F2, 1, D) .* diff_dir + pert_dir;
    
    % Boundary handling with memory
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflected = (lb_rep + repmat(w, 1, D) .* (popdecs - lb_rep)) .* below + ...
                (ub_rep - repmat(w, 1, D) .* (ub_rep - popdecs)) .* above;
    
    offspring = offspring .* (~below & ~above) + reflected;
    
    % Controlled diversity injection
    rand_mask = rand(NP, D) < 0.01 * repmat(1-w, 1, D);
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring = offspring .* ~rand_mask + rand_vals .* rand_mask;
    
    % Final projection with small adaptive perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    offspring = offspring + 0.005*(ub(1)-lb(1))*randn(NP,D).*repmat(1-w,1,D);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
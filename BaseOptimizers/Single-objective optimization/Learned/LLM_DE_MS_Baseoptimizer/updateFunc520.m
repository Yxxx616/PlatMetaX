% MATLAB Code
function [offspring] = updateFunc520(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Enhanced normalization with log scaling
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = log1p(abs_cons) / (log1p(c_max) + eps);
    
    % Improved weight calculation
    w = (1 - norm_fit).^1.8 .* (1 + norm_con).^0.3;
    
    % Elite selection with dynamic priority
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_fit + 0.6*norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Diversity-preserving index selection
    [~, sorted_idx] = sort(rand(NP,1));
    r1 = mod(sorted_idx, NP) + 1;
    r2 = mod(sorted_idx + floor(NP/3), NP) + 1;
    
    % Adaptive mutation parameters
    F1 = 0.9 * w.^1.5;
    F2 = 0.5 * (1 - w).^0.7;
    F3 = 0.3 * sqrt(w);
    sigma = 0.2 * (ub(1) - lb(1)) * (1 - w);
    
    % Vectorized mutation with directional bias
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    cons_factor = repmat(1 + norm_con, 1, D);
    pert_dir = repmat(sigma, 1, D) .* randn(NP, D);
    
    offspring = popdecs + repmat(F1, 1, D) .* elite_dir + ...
                repmat(F2, 1, D) .* diff_dir .* cons_factor + ...
                repmat(F3, 1, D) .* pert_dir;
    
    % Smart boundary handling with memory
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflected = (lb_rep + repmat(w, 1, D) .* (popdecs - lb_rep)) .* below + ...
                (ub_rep - repmat(w, 1, D) .* (ub_rep - popdecs)) .* above;
    
    offspring = offspring .* (~below & ~above) + reflected;
    
    % Adaptive random exploration
    rand_mask = rand(NP, D) < 0.008 * (1 - w);
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring = offspring .* ~rand_mask + rand_vals .* rand_mask;
    
    % Final projection with small perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    offspring = offspring + 0.01 * (ub(1)-lb(1)) * randn(NP, D) .* repmat(1-w, 1, D);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
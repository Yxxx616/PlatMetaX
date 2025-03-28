% MATLAB Code
function [offspring] = updateFunc513(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize fitness and constraints
    norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_con = abs(cons) / (max(abs(cons)) + eps);
    
    % Weight calculations
    w_f = 1 - norm_fit.^0.8;  % Fitness weight (better solutions get higher weight)
    w_c = norm_con.^1.2;      % Constraint weight (more violated gets higher weight)
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        combined = 0.7*norm_con + 0.3*norm_fit;
        [~, elite_idx] = min(combined);
        elite = popdecs(elite_idx,:);
    end
    
    % Random index selection
    idx = randperm(NP);
    r1 = mod(idx, NP) + 1;
    r2 = mod(idx+1, NP) + 1;
    
    % Adaptive scaling factors
    F_base = 0.7 + 0.3 * (1 - w_f);
    F_diff = 0.5 * (1 - w_c) .* rand(NP,1);
    F_pert = 0.3 * w_c .* rand(NP,1);
    
    % Constraint-aware perturbation
    sigma = 0.15 * (1 - sqrt(w_f.*w_c)) .* (ub(1) - lb(1));
    rand_pert = repmat(sigma,1,D) .* randn(NP,D);
    
    % Vectorized mutation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    offspring = popdecs + repmat(F_base,1,D).*elite_dir + ...
                repmat(F_diff,1,D).*rand_dir .* repmat(w_f,1,D) + ...
                repmat(F_pert,1,D).*rand_pert;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflect_factor = 0.3 * repmat(w_f + w_c, 1, D);
    reflected = (lb_rep + reflect_factor.*(lb_rep - offspring)).*below + ...
                (ub_rep - reflect_factor.*(offspring - ub_rep)).*above;
    
    offspring = offspring.*(~below & ~above) + reflected;
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc514(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize fitness and constraints
    norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_con = abs(cons) / (max(abs(cons)) + eps);
    
    % Adaptive weights
    w_f = 1 - norm_fit.^0.5;  % Fitness weight (better solutions get higher weight)
    w_c = norm_con.^0.8;      % Constraint weight (more violated gets higher weight)
    
    % Elite selection with feasibility consideration
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_con + norm_fit);
        elite = popdecs(elite_idx,:);
    end
    
    % Random index selection
    idx = randperm(NP);
    r1 = mod(idx, NP) + 1;
    r2 = mod(idx+1, NP) + 1;
    
    % Adaptive components
    F = 0.5 + 0.3 * (1 - sqrt(w_f.*w_c));
    sigma = 0.2 * (ub(1) - lb(1)) * (1 - w_f.*w_c);
    
    % Vectorized mutation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    pert_dir = repmat(sigma,1,D) .* randn(NP,D);
    
    offspring = popdecs + repmat(F,1,D).*elite_dir + ...
                repmat(0.5 - 0.3*w_c,1,D).*diff_dir + ...
                repmat(0.2*w_f,1,D).*pert_dir;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflect_factor = 0.4 * repmat(w_f + w_c, 1, D);
    reflected = (lb_rep + reflect_factor.*(lb_rep - offspring)).*below + ...
                (ub_rep - reflect_factor.*(offspring - ub_rep)).*above;
    
    offspring = offspring.*(~below & ~above) + reflected;
    offspring = max(min(offspring, ub_rep), lb_rep);
end
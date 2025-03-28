% MATLAB Code
function [offspring] = updateFunc512(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection with combined constraint-fitness metric
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
        norm_con = abs(cons) / (max(abs(cons)) + eps);
        combined = 0.6*norm_con + 0.4*norm_fit;
        [~, elite_idx] = min(combined);
        elite = popdecs(elite_idx,:);
    end
    
    % Adaptive weights calculation
    w_f = ((popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps).^0.9;
    w_c = (abs(cons) ./ (max(abs(cons)) + eps)).^1.5;
    
    % Random index selection
    idx = randperm(NP);
    r1 = mod(idx, NP) + 1;
    r2 = mod(idx+1, NP) + 1;
    
    % Dynamic scaling factors
    F1 = 0.8 + 0.2 * w_f;
    F2 = 0.6 * (1 - w_c) .* rand(NP,1);
    F3 = 0.4 * w_c .* rand(NP,1);
    
    % Constraint-aware perturbation
    sigma = 0.1 * (1 - sqrt(w_f.*w_c)) .* (ub(1) - lb(1));
    rand_pert = repmat(sigma,1,D) .* randn(NP,D);
    
    % Vectorized mutation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    offspring = popdecs + repmat(F1,1,D).*elite_dir + ...
                repmat(F2,1,D).*rand_dir .* repmat(w_f,1,D) + ...
                repmat(F3,1,D).*rand_pert;
    
    % Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflect_factor = 0.25 * repmat(w_f + w_c, 1, D);
    reflected = (lb_rep + reflect_factor.*(lb_rep - offspring)).*below + ...
                (ub_rep - reflect_factor.*(offspring - ub_rep)).*above;
    
    offspring = offspring.*(~below & ~above) + reflected;
    offspring = max(min(offspring, ub_rep), lb_rep);
end
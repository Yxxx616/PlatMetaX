% MATLAB Code
function [offspring] = updateFunc511(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Enhanced elite selection with combined fitness and constraint metric
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        % Normalized metrics
        norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
        norm_con = abs(cons) / (max(abs(cons)) + eps);
        % Combined metric with adaptive weights
        combined = 0.7*norm_con + 0.3*norm_fit;
        [~, elite_idx] = min(combined);
        elite = popdecs(elite_idx, :);
    end
    
    % Adaptive weights calculation
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = ((popfits - min_fit) / (max_fit - min_fit + eps)).^0.8;
    
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    w_c = (abs_cons / (max_con + eps)).^1.2;
    
    % Vectorized random index selection avoiding current index
    [~, idx] = sort(rand(NP, NP), 2);
    mask = idx == (1:NP)';
    idx(mask) = NP;
    r1 = idx(:,1);
    r2 = idx(:,2);
    
    % Dynamic scaling factors
    F1 = 0.7 + 0.3 * rand(NP,1);  % Strong elite guidance
    F2 = 0.5 * (1 - w_f) .* rand(NP,1);
    F3 = 0.3 * (0.1 + 0.9*w_c) .* rand(NP,1);
    
    % Constraint-aware random perturbation
    sigma = 0.05 * (1 - sqrt(w_f.*w_c)) .* (ub(1) - lb(1));
    rand_pert = sigma .* randn(NP,D);
    
    % Vectorized mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    fit_diff = popdecs(r1,:) - popdecs(r2,:);
    
    offspring = popdecs + repmat(F1,1,D).*elite_dir + ...
                repmat(F2,1,D).*fit_diff .* repmat(w_f,1,D) + ...
                repmat(F3,1,D).*rand_pert;
    
    % Enhanced boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflect_factor = 0.3 + 0.7*repmat(sqrt(w_f.*w_c),1,D);
    offspring = offspring.*(~below & ~above) + ...
                (lb_rep + reflect_factor.*(lb_rep - offspring)).*below + ...
                (ub_rep - reflect_factor.*(offspring - ub_rep)).*above;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
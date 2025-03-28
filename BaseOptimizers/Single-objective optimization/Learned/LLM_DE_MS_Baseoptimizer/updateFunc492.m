% MATLAB Code
function [offspring] = updateFunc492(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate fitness weights (inverse fitness proportion)
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_fit = 1 - (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Calculate constraint violation weights
    max_cv = max(abs(cons));
    w_con = abs(cons) / (max_cv + eps);
    
    % Generate random indices (ensure distinct)
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 3));
        r1(i) = selected(1);
        r2(i) = selected(2);
        r3(i) = selected(3);
    end
    
    % Mutation operation with adaptive scaling
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_fit = popdecs(r1,:) - popdecs(r2,:);
    diff_con = popdecs(r2,:) - popdecs(r3,:);
    
    % Apply weights to difference vectors
    diff_fit = diff_fit .* repmat(w_fit, 1, D);
    diff_con = diff_con .* repmat(w_con, 1, D);
    
    % Generate offspring with adaptive components
    F1 = 0.6;
    F2 = 0.3;
    F3 = 0.1;
    F4 = 0.2;
    
    offspring = popdecs + F1.*elite_dir + F2.*diff_fit + F3.*diff_con + ...
                F4*(ub-lb).*randn(NP,D);
    
    % Boundary handling with random reinitialization
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_reinit = lb_rep + (ub_rep-lb_rep).*rand(NP,D);
    offspring = offspring.*(~out_of_bounds) + rand_reinit.*out_of_bounds;
    
    % Final bounds check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
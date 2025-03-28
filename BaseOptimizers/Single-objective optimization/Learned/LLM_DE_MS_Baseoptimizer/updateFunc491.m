% MATLAB Code
function [offspring] = updateFunc491(popdecs, popfits, cons)
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
    
    % Normalize fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    max_cv = max(abs(cons));
    norm_cv = abs(cons) / (max_cv + eps);
    
    % Calculate adaptive scaling factors
    F1 = 0.5 * (1 - norm_cv);
    F2 = 0.3 * norm_fits;
    F3 = 0.2 * norm_cv;
    F4 = 0.1;
    
    % Generate random indices (ensure distinct)
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        selected = candidates(randperm(length(candidates), 4));
        r1(i) = selected(1);
        r2(i) = selected(2);
        r3(i) = selected(3);
        r4(i) = selected(4);
    end
    
    % Mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_vec1 = popdecs(r1,:) - popdecs(r2,:);
    diff_vec2 = popdecs(r3,:) - popdecs(r4,:);
    rand_comp = F4 * (ub - lb) .* randn(NP, D);
    
    offspring = popdecs + F1.*elite_dir + F2.*diff_vec1 + F3.*diff_vec2 + rand_comp;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    offspring = offspring .* ~(lower_violation | upper_violation) + ...
               (2*lb_rep - offspring) .* lower_violation + ...
               (2*ub_rep - offspring) .* upper_violation;
    
    % Ensure final bounds
    offspring = min(max(offspring, lb_rep), ub_rep);
end
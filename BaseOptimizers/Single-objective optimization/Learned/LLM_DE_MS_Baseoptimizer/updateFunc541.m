% MATLAB Code
function [offspring] = updateFunc541(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Fitness-based weights (0=best, 1=worst)
    [~, fit_rank] = sort(popfits);
    w = (fit_rank - 1)' / (NP - 1 + eps);
    
    % Constraint violation weights (0=feasible, 1=most violated)
    abs_cons = abs(cons);
    v = (abs_cons - min(abs_cons)) / (max(abs_cons) - min(abs_cons) + eps);
    
    % Select elite individual (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(abs_cons);
        elite = popdecs(elite_idx,:);
    end
    
    % Generate random indices for difference vectors
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive scaling factors
    F1 = 0.7 * (1 - w .* v);          % Elite guidance
    F2 = 0.5 * (w + v);                % Constraint-aware perturbation
    F3 = 0.3 * (1 - sqrt(w .* v));     % Adaptive exploration
    
    % Mutation operation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    gauss_noise = randn(NP, D) .* repmat(F3', 1, D);
    
    offspring = popdecs + ...
        repmat(F1', 1, D) .* elite_diff + ...
        repmat(F2', 1, D) .* rand_diff .* (1 + repmat(abs(cons), 1, D)) + ...
        gauss_noise;
    
    % Hybrid boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for slight violations
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    % Random reinitialization for severe violations
    severe_mask = (offspring < lb_rep - 0.1*(ub-lb)) | (offspring > ub_rep + 0.1*(ub-lb));
    rand_pos = repmat(lb, NP, 1) + rand(NP, D) .* repmat(ub-lb, NP, 1);
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low .* ~severe_mask + ...
        (2*ub_rep - offspring) .* mask_high .* ~severe_mask + ...
        rand_pos .* severe_mask;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
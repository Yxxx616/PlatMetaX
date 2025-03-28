% MATLAB Code
function [offspring] = updateFunc490(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask, :);
        elite = elite(elite_idx, :);
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
    F1 = 0.8 * (1 - norm_cv);
    F2 = 0.6 * norm_fits;
    F3 = 0.4 * norm_cv;
    
    % Generate random indices (ensure r1 ≠ r2 ≠ i)
    r1 = arrayfun(@(i) setdiff(randperm(NP, 2), 1:NP, 'UniformOutput', false);
    r1 = cell2mat(r1)';
    r2 = arrayfun(@(i) setdiff(randperm(NP, 3), 1:NP, 'UniformOutput', false);
    r2 = cell2mat(r2(:,2))';
    
    % Mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    rand_comp = 0.2 * (ub - lb) .* randn(NP, D);
    
    offspring = popdecs + F1.*elite_dir + F2.*diff_vec + F3.*rand_comp;
    
    % Boundary handling with random reinitialization
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_pos = lb_rep + (ub_rep - lb_rep) .* rand(NP, D);
    offspring = offspring .* ~out_of_bounds + rand_pos .* out_of_bounds;
end
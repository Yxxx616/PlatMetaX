% MATLAB Code
function [offspring] = updateFunc488(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify feasible and infeasible solutions
    feasible_mask = cons <= 0;
    feasible_count = sum(feasible_mask);
    
    % Select elite solution
    if feasible_count > 0
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask, :);
        elite = elite(elite_idx, :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize constraint violations
    min_cv = min(cons);
    max_cv = max(cons);
    norm_cv = (cons - min_cv) / (max_cv - min_cv + eps);
    
    % Prepare indices for feasible and infeasible solutions
    feasible_indices = find(feasible_mask);
    infeasible_indices = find(~feasible_mask);
    
    % Generate random indices
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    r3 = zeros(NP, 1);
    r4 = zeros(NP, 1);
    
    for i = 1:NP
        if feasible_count > 1
            temp = feasible_indices(randperm(feasible_count, 2));
            r1(i) = temp(1);
            r2(i) = temp(2);
        else
            r1(i) = randi(NP);
            r2(i) = randi(NP);
        end
        
        if (NP - feasible_count) > 1
            temp = infeasible_indices(randperm(NP - feasible_count, 2));
            r3(i) = temp(1);
            r4(i) = temp(2);
        else
            r3(i) = randi(NP);
            r4(i) = randi(NP);
        end
    end
    
    % Calculate scaling factors
    F1 = 0.5 * (1 + norm_cv);
    F2 = 0.5 * (1 - norm_cv);
    F3 = 0.3 * norm_cv;
    
    % Mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_feasible = popdecs(r1,:) - popdecs(r2,:);
    diff_infeasible = popdecs(r3,:) - popdecs(r4,:);
    
    offspring = popdecs + F1.*elite_dir + F2.*diff_feasible + F3.*diff_infeasible;
    
    % Boundary handling with random reinitialization
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_out = (offspring < lb_rep) | (offspring > ub_rep);
    rand_vals = lb_rep + (ub_rep - lb_rep) .* rand(NP, D);
    offspring = offspring .* ~mask_out + rand_vals .* mask_out;
end
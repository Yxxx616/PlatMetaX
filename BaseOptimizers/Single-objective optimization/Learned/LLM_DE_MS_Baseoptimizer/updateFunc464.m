% MATLAB Code
function [offspring] = updateFunc464(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility analysis
    feasible_mask = cons <= 0;
    alpha = sum(feasible_mask) / NP;
    
    % Elite selection
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Feasible center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
    end
    
    % Normalized constraints
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive scaling factors
    F_elite = 0.4 * (1 - alpha) * (1 + rand(NP, 1));
    F_center = 0.3 * alpha * (1 + rand(NP, 1));
    F_diff = 0.5 * alpha * (1 - norm_cons);
    F_rand = 0.6 * (1 - alpha) * norm_cons;
    
    % Expand to D dimensions
    F_elite = repmat(F_elite, 1, D);
    F_center = repmat(F_center, 1, D);
    F_diff = repmat(F_diff, 1, D);
    F_rand = repmat(F_rand, 1, D);
    
    % Random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)');
    while any(invalid)
        r1(invalid) = randi(NP, sum(invalid), 1);
        r2(invalid) = randi(NP, sum(invalid), 1);
        r3(invalid) = randi(NP, sum(invalid), 1);
        invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)');
    end
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    center_dir = repmat(feas_center, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    rand_dir = popdecs(r3,:) - popdecs;
    
    % Combined mutation
    offspring = popdecs + F_elite.*elite_dir + F_center.*center_dir + ...
                F_diff.*diff_dir + F_rand.*rand_dir;
    
    % Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Identify out-of-bounds dimensions
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    % Random reinitialization or reflection
    rand_mask = rand(NP,D) < 0.5;
    reflect_low = mask_low & ~rand_mask;
    reflect_high = mask_high & ~rand_mask;
    
    % Apply corrections
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (lb_rep + (ub_rep-lb_rep).*rand(NP,D)) .* (mask_low | mask_high) .* rand_mask + ...
               (2*lb_rep - offspring) .* reflect_low + ...
               (2*ub_rep - offspring) .* reflect_high;
end
% MATLAB Code
function [offspring] = updateFunc469(popdecs, popfits, cons)
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
    
    % Feasible center calculation
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
    end
    
    % Normalized constraints with quadratic transformation
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Enhanced adaptive scaling factors
    F_elite = 0.5 * (1 - alpha) * (0.4 + 0.6*randn(NP, 1));
    F_center = 0.3 * alpha * (0.5 + 0.5*rand(NP, 1));
    F_con = 0.4 * (1 - norm_cons).^2;
    F_rand = 0.3 * norm_cons.^1.5;
    
    % Expand to D dimensions
    F_elite = repmat(F_elite, 1, D);
    F_center = repmat(F_center, 1, D);
    F_con = repmat(F_con, 1, D);
    F_rand = repmat(F_rand, 1, D);
    
    % Random indices selection with guaranteed distinctness
    r1 = randi(NP, NP, 1);
    r2 = arrayfun(@(x) setdiff(randperm(NP), x)', 1:NP, 'UniformOutput', false);
    r2 = cell2mat(r2)';
    r2 = r2(:,1);
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    center_dir = repmat(feas_center, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation with enhanced balance
    offspring = popdecs + F_elite.*elite_dir + F_center.*center_dir + ...
                F_con.*diff_dir + F_rand.*(popdecs(randperm(NP),:) - popdecs);
    
    % Boundary handling with vectorized operations
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability
    reflect_prob = 0.8;
    reflect_mask = rand(NP,D) < reflect_prob;
    
    % Boundary handling
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* (mask_low & reflect_mask) + ...
               (2*ub_rep - offspring) .* (mask_high & reflect_mask) + ...
               (lb_rep + (ub_rep-lb_rep).*rand(NP,D)) .* (mask_low | mask_high) .* ~reflect_mask;
    
    % Ensure numerical stability
    offspring = max(min(offspring, ub_rep), lb_rep);
end
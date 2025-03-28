% MATLAB Code
function [offspring] = updateFunc971(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    v = abs(cons);
    v_max = max(v) + eps;
    
    % Weight factors
    w_f = (popfits - f_min)./f_range;
    w_c = 1 - v./v_max;
    
    % Identify elite solution (minimum fitness with minimum constraint violation)
    [~, elite_idx] = min(popfits + 1000*v);
    x_elite = popdecs(elite_idx,:);
    
    % Generate random indices (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx ~= (1:NP)';
    rand_idx = rand_idx .* mask;
    rand_idx(rand_idx == 0) = 1;
    r = rand_idx(:,1:2);
    
    % Adaptive scaling factors
    F1 = 0.8 * w_c;
    F2 = 0.6 * w_f;
    
    % Elite guidance vectors
    elite_diff = x_elite(ones(NP,1),:) - popdecs;
    
    % Random difference vectors
    diff = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    
    % Constraint-aware opposition vectors
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    opposition = lb_matrix + ub_matrix - popdecs;
    
    % Hybrid mutation
    mutants = popdecs + F1(:, ones(1, D)).*elite_diff + ...
              F2(:, ones(1, D)).*diff + ...
              0.4 * w_c(:, ones(1, D)).*opposition;
    
    % Dynamic crossover rate
    CR = 0.9 * w_c + 0.1;
    
    % Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_violation = offspring < lb_matrix;
    ub_violation = offspring > ub_matrix;
    
    % For lower bound violations
    overshoot = lb_matrix(lb_violation) - offspring(lb_violation);
    offspring(lb_violation) = lb_matrix(lb_violation) + 0.25 * overshoot;
    
    % For upper bound violations
    overshoot = offspring(ub_violation) - ub_matrix(ub_violation);
    offspring(ub_violation) = ub_matrix(ub_violation) - 0.25 * overshoot;
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
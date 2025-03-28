% MATLAB Code
function [offspring] = updateFunc969(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    v = abs(cons);
    v_max = max(v) + eps;
    
    % Identify elite solution (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1000*v);
    x_elite = popdecs(elite_idx,:);
    
    % Generate random indices (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx ~= (1:NP)';
    rand_idx = rand_idx .* mask;
    rand_idx(rand_idx == 0) = 1;
    r = rand_idx(:,1:2);
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - v./v_max).^0.5;
    F2 = 0.6 * (1 - (popfits - f_min)./f_range).^0.3;
    
    % Elite guidance vectors
    elite_diff = x_elite - popdecs;
    
    % Random difference vectors
    diff = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    
    % Constrained opposition vectors
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    opposition = lb_matrix + ub_matrix - popdecs .* (1 - v./v_max);
    
    % Hybrid mutation
    mutants = popdecs + F1(:, ones(1, D)).*elite_diff + ...
              F2(:, ones(1, D)).*diff + 0.4*(opposition - popdecs);
    
    % Dynamic crossover rate
    CR = 0.9 * (1 - v./v_max).^0.2 + 0.1;
    
    % Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Adaptive boundary handling
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
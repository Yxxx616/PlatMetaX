% MATLAB Code
function [offspring] = updateFunc968(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    v_max = max(abs(cons)) + eps;
    v = abs(cons);
    
    % Compute fitness similarity weights
    f_diff = abs(popfits - popfits');
    sigma_f = std(popfits) + eps;
    W = exp(-f_diff./(sigma_f + eps));
    
    % Compute constraint violation weights
    v_sum = v + v';
    sigma_v = std(v) + eps;
    C = 1 - min(1, v_sum./(2*sigma_v));
    
    % Identify elite solution (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1000*v);
    x_elite = popdecs(elite_idx,:);
    
    % Generate random indices (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx ~= (1:NP)';
    rand_idx = rand_idx .* mask;
    rand_idx(rand_idx == 0) = 1;
    r = rand_idx(:,1:4);
    
    % Adaptive scaling factors
    F1 = 0.7 * (1 - v./v_max).^0.3;
    F2 = 0.5 * (1 - (popfits - f_min)./f_range).^0.4;
    F3 = 0.3 * (1 - v./v_max).^0.2;
    
    % Enhanced mutation with weighted differences
    elite_diff = x_elite - popdecs;
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    % Apply weights (vectorized)
    W_sel = W(sub2ind([NP, NP], (1:NP)', r(:,1)));
    C_sel = C(sub2ind([NP, NP], (1:NP)', r(:,3)));
    diff1 = diff1 .* W_sel(:, ones(1, D));
    diff2 = diff2 .* C_sel(:, ones(1, D));
    
    mutants = popdecs + F1(:, ones(1, D)).*elite_diff + ...
              F2(:, ones(1, D)).*diff1 + F3(:, ones(1, D)).*diff2;
    
    % Dynamic crossover rate based on constraint violation
    CR = 0.85 * (1 - v./v_max).^0.25 + 0.15;
    
    % Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Handle lower bound violations
    below_lb = offspring < lb_matrix;
    overshoot = lb_matrix(below_lb) - offspring(below_lb);
    offspring(below_lb) = lb_matrix(below_lb) + 0.3 * overshoot;
    
    % Handle upper bound violations
    above_ub = offspring > ub_matrix;
    overshoot = offspring(above_ub) - ub_matrix(above_ub);
    offspring(above_ub) = ub_matrix(above_ub) - 0.3 * overshoot;
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
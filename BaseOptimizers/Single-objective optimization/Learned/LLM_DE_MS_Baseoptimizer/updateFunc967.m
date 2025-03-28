% MATLAB Code
function [offspring] = updateFunc967(popdecs, popfits, cons)
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
    W = exp(-f_diff/sigma_f);
    
    % Compute constraint violation weights
    v_sum = v + v';
    sigma_v = std(v) + eps;
    C = 1 - min(1, v_sum/(2*sigma_v));
    
    % Identify elite solution
    [~, elite_idx] = min(popfits + 1000*v);
    x_elite = popdecs(elite_idx,:);
    
    % Generate random indices
    r = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 4));
    end
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - v/v_max).^0.5;
    F2 = 0.6 * (1 - (popfits - f_min)/f_range);
    F3 = 0.4 * (1 - v/v_max);
    
    % Enhanced mutation with fitness and constraint awareness
    elite_diff = x_elite - popdecs;
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    % Apply weights
    for i = 1:NP
        w = W(r(i,1), r(i,2));
        c = C(r(i,3), r(i,4));
        diff1(i,:) = diff1(i,:) * w;
        diff2(i,:) = diff2(i,:) * c;
    end
    
    mutants = popdecs + F1.*elite_diff + F2.*diff1 + F3.*diff2;
    
    % Dynamic crossover rate
    CR = 0.9 * (1 - v/v_max).^0.2 + 0.1;
    
    % Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Lower bound violations
    below_lb = offspring < lb_matrix;
    offspring(below_lb) = lb_matrix(below_lb) + 0.5*(lb_matrix(below_lb) - offspring(below_lb));
    
    % Upper bound violations
    above_ub = offspring > ub_matrix;
    offspring(above_ub) = ub_matrix(above_ub) - 0.5*(offspring(above_ub) - ub_matrix(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
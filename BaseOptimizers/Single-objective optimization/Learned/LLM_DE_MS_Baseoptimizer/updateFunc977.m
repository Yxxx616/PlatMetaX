% MATLAB Code
function [offspring] = updateFunc977(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness (assuming maximization)
    f_min = min(popfits);
    f_max = max(popfits);
    f_norm = (popfits - f_min) ./ (f_max - f_min + eps);
    
    % Normalize constraints
    c_abs = abs(cons);
    c_max = max(c_abs);
    c_norm = c_abs ./ (c_max + eps);
    
    % Adaptive parameters
    F = 0.5;  % Base scaling factor
    alpha = 0.5 * (1 - tanh(c_norm));  % Constraint influence
    w = 1 ./ (1 + exp(-5*f_norm));     % Fitness weights
    
    % Select elite solutions (top 30%)
    [~, idx] = sort(popfits, 'descend');
    elite = popdecs(idx(1:ceil(NP*0.3)),:);
    x_elite = mean(elite, 1);
    
    % Generate random indices
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r = available(randperm(length(available), 2));
        r1(i) = r(1);
        r2(i) = r(2);
    end
    
    % Mutation components
    elite_term = x_elite(ones(NP,1),:) - popdecs;
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    cons_term = sign(cons) .* randn(NP, D);
    
    % Combined mutation
    mutants = popdecs + F*elite_term + ...
              alpha(:, ones(1, D)).*cons_term + ...
              w(:, ones(1, D)).*diff_term;
    
    % Adaptive crossover rate
    CR = 0.3 + 0.5 * (1 - tanh(c_norm));
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with fitness-based reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Lower bound violations
    lb_viol = offspring < lb_matrix;
    overshoot = lb_matrix(lb_viol) - offspring(lb_viol);
    offspring(lb_viol) = lb_matrix(lb_viol) + w(lb_viol(:,1), ones(1,sum(lb_viol(1,:)))).*overshoot;
    
    % Upper bound violations
    ub_viol = offspring > ub_matrix;
    overshoot = offspring(ub_viol) - ub_matrix(ub_viol);
    offspring(ub_viol) = ub_matrix(ub_viol) - w(ub_viol(:,1), ones(1,sum(ub_viol(1,:)))).*overshoot;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
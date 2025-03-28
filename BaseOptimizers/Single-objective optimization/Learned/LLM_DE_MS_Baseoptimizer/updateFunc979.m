% MATLAB Code
function [offspring] = updateFunc979(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness (assuming minimization)
    f_min = min(popfits);
    f_max = max(popfits);
    f_norm = (popfits - f_min) / (f_max - f_min + eps);
    
    % Normalize constraints
    c_abs = abs(cons);
    c_norm = c_abs / (max(c_abs) + eps);
    
    % Adaptive parameters
    F = 0.9 - 0.5 * (979/1000); % Linearly decreasing F
    alpha = 0.5 * (1 - tanh(c_norm));
    beta = 0.3 * tanh(c_norm);
    
    % Select elite solutions (top 20%)
    [~, idx] = sort(popfits);
    elite = popdecs(idx(1:ceil(NP*0.2)),:);
    x_elite = mean(elite, 1);
    
    % Generate random indices (4 per individual)
    r = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 4));
    end
    
    % Compute fitness weights using softmax
    f_exp = exp(-popfits(r));
    w = f_exp ./ sum(f_exp, 2);
    
    % Mutation components
    elite_term = x_elite(ones(NP,1),:) - popdecs;
    diff_term = w(:,1).*(popdecs(r(:,1),:) - popdecs(r(:,2),:)) + ...
                w(:,2).*(popdecs(r(:,3),:) - popdecs(r(:,4),:));
    cons_term = sign(cons) .* randn(NP, D);
    
    % Combined mutation
    mutants = popdecs + F*elite_term + ...
              alpha(:, ones(1, D)).*diff_term + ...
              beta(:, ones(1, D)).*cons_term;
    
    % Adaptive crossover rate
    CR = 0.5 + 0.4*(1 - tanh(c_norm));
    
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
    offspring(lb_viol) = lb_matrix(lb_viol) + (1-f_norm(lb_viol(:,1), ones(1,sum(lb_viol(1,:)))).*overshoot;
    
    % Upper bound violations
    ub_viol = offspring > ub_matrix;
    overshoot = offspring(ub_viol) - ub_matrix(ub_viol);
    offspring(ub_viol) = ub_matrix(ub_viol) - (1-f_norm(ub_viol(:,1), ones(1,sum(ub_viol(1,:)))).*overshoot;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
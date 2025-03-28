% MATLAB Code
function [offspring] = updateFunc974(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_norm = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    c_norm = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Compute adaptive weights (fitness importance)
    w = 1 ./ (1 + exp(-5*f_norm));
    
    % Select elite solutions (top 20%)
    [~, idx] = sort(w, 'descend');
    elite = popdecs(idx(1:ceil(NP*0.2)),:);
    x_elite = mean(elite, 1);
    
    % Generate random indices for differential term
    r = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 2));
    end
    
    % Adaptive scaling factors
    F = 0.5 + 0.4 * tanh(f_norm);
    alpha = 0.1 * exp(-c_norm);
    
    % Mutation components
    elite_term = x_elite(ones(NP,1),:) - popdecs;
    diff_term = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    cons_term = sign(popdecs) .* c_norm(:, ones(1, D));
    
    % Combined mutation
    mutants = popdecs + F(:, ones(1, D)).*elite_term + ...
              (1-F(:, ones(1, D))).*diff_term + ...
              alpha(:, ones(1, D)).*cons_term;
    
    % Dynamic crossover rate based on constraints
    CR = 0.9 * (1 - tanh(c_norm)) + 0.1;
    
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
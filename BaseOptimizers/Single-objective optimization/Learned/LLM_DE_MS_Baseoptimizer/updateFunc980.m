% MATLAB Code
function [offspring] = updateFunc980(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    
    % Adaptive parameters
    F = 0.9 - 0.5 * (980/1000); % Decreasing F
    alpha = 0.5 * (1 - tanh(c_norm));
    beta = 0.3 * tanh(c_norm);
    
    % Select top 30% as elite
    [~, idx] = sort(popfits);
    elite = popdecs(idx(1:ceil(NP*0.3)),:);
    x_elite = mean(elite, 1);
    elite_term = x_elite(ones(NP,1),:) - popdecs;
    
    % Generate random indices (4 per individual)
    r = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 4));
    end
    
    % Fitness-weighted differential term
    f_weights = exp(-popfits(r));
    f_weights = f_weights ./ sum(f_weights, 2);
    diff_term = f_weights(:,1).*(popdecs(r(:,1),:) - popdecs(r(:,2),:)) + ...
                f_weights(:,2).*(popdecs(r(:,3),:) - popdecs(r(:,4),:));
    
    % Constraint-driven perturbation
    cons_term = tanh(abs(cons)) .* randn(NP, D);
    
    % Combined mutation
    mutants = popdecs + F*elite_term + ...
              alpha(:, ones(1, D)).*diff_term + ...
              beta(:, ones(1, D)).*cons_term;
    
    % Adaptive crossover
    CR = 0.5 + 0.4*(1 - tanh(c_norm));
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Fitness-based boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Reflect with fitness-based probability
    lb_viol = offspring < lb_matrix;
    offspring(lb_viol) = lb_matrix(lb_viol) + (1-f_norm(lb_viol(:,1))).*abs(offspring(lb_viol)-lb_matrix(lb_viol));
    
    ub_viol = offspring > ub_matrix;
    offspring(ub_viol) = ub_matrix(ub_viol) - (1-f_norm(ub_viol(:,1))).*abs(offspring(ub_viol)-ub_matrix(ub_viol)));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
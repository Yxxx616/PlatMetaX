% MATLAB Code
function [offspring] = updateFunc985(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps;
    c_norm = cons / (max(abs(cons)) + eps);
    
    % Select top 20% as elite
    [~, idx] = sort(popfits);
    elite_size = max(2, ceil(NP*0.2));
    elite = popdecs(idx(1:elite_size),:);
    x_elite = mean(elite, 1);
    
    % Fitness weights
    weights = exp(-popfits);
    weights = weights / sum(weights);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(f_norm);
    alpha = 0.5 * (1 + tanh(c_norm));
    CR = 0.9 - 0.5 * abs(c_norm);
    
    % Generate random indices (vectorized)
    r = zeros(NP, 8);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 8));
    end
    
    % Weighted differential term
    diff_term = weights(r(:,1)) .* (popdecs(r(:,1),:) - popdecs(r(:,5),:)) + ...
               weights(r(:,2)) .* (popdecs(r(:,2),:) - popdecs(r(:,6),:)) + ...
               weights(r(:,3)) .* (popdecs(r(:,3),:) - popdecs(r(:,7),:)) + ...
               weights(r(:,4)) .* (popdecs(r(:,4),:) - popdecs(r(:,8),:));
    
    % Elite direction term
    elite_term = x_elite(ones(NP,1),:) - popdecs;
    
    % Mutation
    mutants = popdecs + F.*elite_term + alpha.*diff_term;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with fitness-based reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    reflect_prob = 1 - f_norm;  % Better solutions reflect more
    lb_viol = offspring < lb_matrix;
    offspring(lb_viol) = lb_matrix(lb_viol) + reflect_prob(lb_viol(:,1)).*(ub_matrix(lb_viol)-lb_matrix(lb_viol)).*rand(sum(lb_viol(:)),1);
    
    ub_viol = offspring > ub_matrix;
    offspring(ub_viol) = ub_matrix(ub_viol) - reflect_prob(ub_viol(:,1)).*(ub_matrix(ub_viol)-lb_matrix(ub_viol)).*rand(sum(ub_viol(:)),1);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
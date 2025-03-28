% MATLAB Code
function [offspring] = updateFunc986(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    c_norm = cons / (max(abs(cons)) + eps);
    
    % Select top 20% as elite
    [~, idx] = sort(popfits);
    elite_size = max(2, ceil(NP*0.2));
    elite = popdecs(idx(1:elite_size),:);
    x_elite = mean(elite, 1);
    
    % Fitness weights (softmax)
    weights = exp(-popfits - min(-popfits)); % Numerical stability
    weights = weights / sum(weights);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(f_norm);
    alpha = 0.5 * (1 + tanh(c_norm));
    CR = 0.9 - 0.5 * abs(c_norm);
    
    % Generate random indices (vectorized)
    r1 = zeros(NP, 4);
    r2 = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 8));
        r1(i,:) = ridx(1:4);
        r2(i,:) = ridx(5:8);
    end
    
    % Weighted differential term
    diff_term = weights(r1(:,1)) .* (popdecs(r1(:,1),:) - popdecs(r2(:,1),:)) + ...
               weights(r1(:,2)) .* (popdecs(r1(:,2),:) - popdecs(r2(:,2),:)) + ...
               weights(r1(:,3)) .* (popdecs(r1(:,3),:) - popdecs(r2(:,3),:)) + ...
               weights(r1(:,4)) .* (popdecs(r1(:,4),:) - popdecs(r2(:,4),:));
    
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
    ub_viol = offspring > ub_matrix;
    
    % Reflection for lower bound violations
    refl_range = ub_matrix - lb_matrix;
    offspring(lb_viol) = lb_matrix(lb_viol) + ...
        reflect_prob(lb_viol(:,1), ones(1,D)).*refl_range(lb_viol).*rand(sum(lb_viol(:)),1);
    
    % Reflection for upper bound violations
    offspring(ub_viol) = ub_matrix(ub_viol) - ...
        reflect_prob(ub_viol(:,1), ones(1,D)).*refl_range(ub_viol).*rand(sum(ub_viol(:)),1);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
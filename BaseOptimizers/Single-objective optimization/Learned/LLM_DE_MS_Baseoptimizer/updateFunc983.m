% MATLAB Code
function [offspring] = updateFunc983(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_norm = popfits / (max(abs(popfits)) + eps);
    c_norm = cons / (max(abs(cons)) + eps);
    
    % Select top 30% as elite
    [~, idx] = sort(popfits);
    elite_size = ceil(NP*0.3);
    elite = popdecs(idx(1:elite_size),:);
    x_elite = mean(elite, 1);
    
    % Adaptive parameters
    F = 0.5 * (1 + tanh(f_norm));
    alpha = 0.3 * (1 - exp(-abs(c_norm)));
    beta = 0.2 * ones(NP,1);
    
    % Generate 8 random indices per individual (vectorized)
    r = zeros(NP, 8);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 8));
    end
    
    % Fitness-weighted differential term
    f1 = popfits(r(:,1:4)); f2 = popfits(r(:,5:8));
    weights = exp(-f1-f2);
    weights = weights ./ sum(weights, 2);
    diff_term = weights(:,1).*(popdecs(r(:,1),:) - popdecs(r(:,5),:)) + ...
               weights(:,2).*(popdecs(r(:,2),:) - popdecs(r(:,6),:)) + ...
               weights(:,3).*(popdecs(r(:,3),:) - popdecs(r(:,7),:)) + ...
               weights(:,4).*(popdecs(r(:,4),:) - popdecs(r(:,8),:));
    
    % Constraint-driven perturbation
    cons_term = sign(c_norm) .* randn(NP, D) .* (1 - exp(-abs(c_norm)));
    
    % Elite direction
    elite_term = x_elite(ones(NP,1),:) - popdecs;
    
    % Combined mutation
    mutants = popdecs + F.*elite_term + ...
              alpha.*diff_term + ...
              beta.*cons_term;
    
    % Adaptive crossover
    CR = 0.9 - 0.5*abs(c_norm);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with fitness-based reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Reflect based on fitness
    reflect_prob = 1 - f_norm;
    lb_viol = offspring < lb_matrix;
    offspring(lb_viol) = lb_matrix(lb_viol) + reflect_prob(lb_viol(:,1)).*abs(offspring(lb_viol)-lb_matrix(lb_viol));
    
    ub_viol = offspring > ub_matrix;
    offspring(ub_viol) = ub_matrix(ub_viol) - reflect_prob(ub_viol(:,1)).*abs(offspring(ub_viol)-ub_matrix(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
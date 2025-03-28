% MATLAB Code
function [offspring] = updateFunc987(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Compute weights combining fitness and constraints
    sigma_f = std(popfits) + eps;
    sigma_c = std(cons) + eps;
    weights = exp(-popfits/sigma_f - cons/sigma_c);
    weights = weights / sum(weights);
    
    % Weighted elite center
    elite = sum(popdecs .* weights, 1) ./ sum(weights);
    
    % Generate random indices (vectorized)
    r1 = zeros(NP, 3);
    r2 = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 6));
        r1(i,:) = ridx(1:3);
        r2(i,:) = ridx(4:6);
    end
    
    % Adaptive differential terms with Gaussian weights
    alpha = 0.5 * randn(NP, 3);
    diff_terms = alpha(:,1) .* (popdecs(r1(:,1),:) - popdecs(r2(:,1),:)) + ...
                 alpha(:,2) .* (popdecs(r1(:,2),:) - popdecs(r2(:,2),:)) + ...
                 alpha(:,3) .* (popdecs(r1(:,3),:) - popdecs(r2(:,3),:));
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(popfits);
    beta = 0.5 * (1 + tanh(cons));
    
    % Mutation
    elite_term = elite(ones(NP,1),:) - popdecs;
    mutants = popdecs + F.*elite_term + beta.*diff_terms;
    
    % Crossover with adaptive CR
    CR = 0.9 - 0.5 * abs(cons);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    reflect_prob = 1 - abs(cons)/max(abs(cons)+eps);
    lb_viol = offspring < lb_matrix;
    ub_viol = offspring > ub_matrix;
    
    % Reflection for violations
    refl_range = ub_matrix - lb_matrix;
    offspring(lb_viol) = lb_matrix(lb_viol) + ...
        reflect_prob(lb_viol(:,1), ones(1,D)).*refl_range(lb_viol).*rand(sum(lb_viol(:)),1);
    offspring(ub_viol) = ub_matrix(ub_viol) - ...
        reflect_prob(ub_viol(:,1), ones(1,D)).*refl_range(ub_viol).*rand(sum(ub_viol(:)),1);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
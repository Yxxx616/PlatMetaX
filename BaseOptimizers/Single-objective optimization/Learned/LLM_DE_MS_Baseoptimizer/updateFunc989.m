% MATLAB Code
function [offspring] = updateFunc989(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    c_avg = mean(abs(cons)) + eps;
    c_max = max(abs(cons)) + eps;
    f_min = min(popfits);
    f_range = max(popfits) - f_min + eps;
    
    norm_cons = abs(cons) / c_max;
    norm_fits = (popfits - f_min) / f_range;
    
    % Identify best solution (considering both fitness and constraints)
    [~, best_idx] = min(norm_cons + norm_fits);
    x_best = popdecs(best_idx, :);
    
    % Calculate weighted centroid
    weights = exp(-norm_cons - norm_fits);
    weights = weights / sum(weights);
    centroid = sum(popdecs .* weights, 1);
    
    % Generate random indices (vectorized)
    r1 = zeros(NP, 3);
    r2 = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 6));
        r1(i,:) = ridx(1:3);
        r2(i,:) = ridx(4:6);
    end
    
    % Constraint-aware differential weights
    c_r1 = norm_cons(r1);
    c_r2 = norm_cons(r2);
    diff_weights = exp(-(c_r1 + c_r2)/2);
    diff_weights = diff_weights ./ sum(diff_weights, 2);
    
    % Adaptive scaling factors
    F1 = 0.5 * (1 + tanh(norm_fits));
    F2 = 0.5 * (1 - tanh(norm_cons));
    F3 = 0.3 * (1 - exp(-norm_cons/c_avg));
    
    % Mutation components
    elite_term = x_best - popdecs;
    diff_terms = sum((popdecs(r1,:) - popdecs(r2,:)) .* diff_weights(:,[1 1 1]), 2);
    centroid_term = centroid - popdecs;
    
    % Combined mutation
    mutants = popdecs + F1.*elite_term + F2.*diff_terms + F3.*centroid_term;
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    reflect_prob = 1 - norm_cons;
    lb_viol = offspring < lb_matrix;
    ub_viol = offspring > ub_matrix;
    
    offspring(lb_viol) = lb_matrix(lb_viol) + ...
        reflect_prob(lb_viol(:,1)).*rand(sum(lb_viol(:)),1).*(ub_matrix(lb_viol)-lb_matrix(lb_viol));
    offspring(ub_viol) = ub_matrix(ub_viol) - ...
        reflect_prob(ub_viol(:,1)).*rand(sum(ub_viol(:)),1).*(ub_matrix(ub_viol)-lb_matrix(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
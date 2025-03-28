% MATLAB Code
function [offspring] = updateFunc988(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    sigma_c = std(cons) + eps;
    sigma_f = std(popfits) + eps;
    norm_cons = abs(cons) / sigma_c;
    norm_fits = (popfits - min(popfits)) / sigma_f;
    
    % Identify best solution (least infeasible or best fitness)
    [~, best_idx] = min(norm_cons + norm_fits);
    x_best = popdecs(best_idx, :);
    
    % Weighted centroid calculation
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
    diff_weights = exp(-norm_cons(r1) - norm_cons(r2));
    diff_weights = diff_weights ./ sum(diff_weights, 2);
    
    % Mutation components
    elite_term = centroid - popdecs;
    diff_terms = sum((popdecs(r1,:) - popdecs(r2,:)) .* diff_weights(:,[1 1 1]), 2);
    pert_term = x_best - popdecs;
    
    % Adaptive parameters
    F = 0.5 * (1 + tanh(norm_fits));
    beta = 0.5 * (1 - tanh(norm_cons));
    gamma = 0.3 * (1 - exp(-norm_cons));
    
    % Combined mutation
    mutants = popdecs + F.*elite_term + beta.*diff_terms + gamma.*pert_term;
    
    % Adaptive crossover
    CR = 0.9 - 0.4 * norm_cons / max(norm_cons);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    reflect_prob = 1 - norm_cons / max(norm_cons);
    lb_viol = offspring < lb_matrix;
    ub_viol = offspring > ub_matrix;
    
    offspring(lb_viol) = lb_matrix(lb_viol) + ...
        reflect_prob(lb_viol(:,1)).*rand(sum(lb_viol(:)),1).*(ub_matrix(lb_viol)-lb_matrix(lb_viol));
    offspring(ub_viol) = ub_matrix(ub_viol) - ...
        reflect_prob(ub_viol(:,1)).*rand(sum(ub_viol(:)),1).*(ub_matrix(ub_viol)-lb_matrix(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
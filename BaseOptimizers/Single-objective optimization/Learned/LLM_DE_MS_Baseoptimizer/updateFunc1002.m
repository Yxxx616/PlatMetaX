% MATLAB Code
function [offspring] = updateFunc1002(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    norm_cons = tanh(abs(cons));
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Select best solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feasible_indices = find(feasible_mask);
        x_best = popdecs(feasible_indices(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % Generate random indices for differential terms
    r1 = zeros(NP, 3);
    r2 = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 6));
        r1(i,:) = ridx(1:3);
        r2(i,:) = ridx(4:6);
    end
    
    % Constraint-aware weights
    cons_r1 = abs(cons(r1));
    cons_r2 = abs(cons(r2));
    weights = exp(-(cons_r1 + cons_r2));
    weights = weights ./ sum(weights, 2);
    
    % Weighted differential terms
    diff_terms = zeros(NP, D);
    for k = 1:3
        diff_terms = diff_terms + weights(:,k) .* (popdecs(r1(:,k),:) - popdecs(r2(:,k),:));
    end
    
    % Adaptive scaling factors
    F_f = 0.5 + 0.3 * (1 - norm_fits);
    F_c = 0.4 + 0.4 * norm_cons;
    F = F_f .* F_c;
    
    % Constraint-driven perturbation
    perturbation = norm_cons .* 0.1 .* randn(NP, 1) .* ones(1, D);
    
    % Mutation
    elite_term = bsxfun(@minus, x_best, popdecs);
    mutants = popdecs + F.*elite_term + (1-F).*diff_terms + perturbation;
    
    % Adaptive crossover
    CR = 0.7 + 0.2 * (1 - norm_fits) .* (1 - norm_cons);
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
    
    % Apply reflection with probability based on constraint violation
    reflect_mask_lb = lb_viol & (rand(NP,D) < reflect_prob(:,ones(1,D)));
    reflect_mask_ub = ub_viol & (rand(NP,D) < reflect_prob(:,ones(1,D)));
    
    % For non-reflected violations, generate random values
    rand_mask_lb = lb_viol & ~reflect_mask_lb;
    rand_mask_ub = ub_viol & ~reflect_mask_ub;
    
    % Handle reflections
    offspring(reflect_mask_lb) = 2*lb_matrix(reflect_mask_lb) - offspring(reflect_mask_lb);
    offspring(reflect_mask_ub) = 2*ub_matrix(reflect_mask_ub) - offspring(reflect_mask_ub);
    
    % Handle random replacements
    offspring(rand_mask_lb) = lb_matrix(rand_mask_lb) + rand(sum(rand_mask_lb(:)),1).*(ub_matrix(rand_mask_lb)-lb_matrix(rand_mask_lb));
    offspring(rand_mask_ub) = lb_matrix(rand_mask_ub) + rand(sum(rand_mask_ub(:)),1).*(ub_matrix(rand_mask_ub)-lb_matrix(rand_mask_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
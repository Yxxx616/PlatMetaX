% MATLAB Code
function [offspring] = updateFunc1004(popdecs, popfits, cons)
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
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 2));
        r1(i) = ridx(1);
        r2(i) = ridx(2);
    end
    
    % Adaptive scaling factors
    F = 0.4 + 0.3*(1 - norm_fits) + 0.3*norm_cons;
    
    % Hybrid mutation with adaptive perturbation
    elite_term = bsxfun(@minus, x_best, popdecs);
    diff_terms = popdecs(r1,:) - popdecs(r2,:);
    perturbation = norm_cons .* 0.1 .* randn(NP, 1) .* ones(1, D);
    mutants = popdecs + F.*elite_term + (1-F).*diff_terms + perturbation;
    
    % Constraint-aware crossover
    CR = 0.9 - 0.4*norm_fits + 0.2*norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Advanced boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    reflect_prob = 1 - norm_cons;
    lb_viol = offspring < lb_matrix;
    ub_viol = offspring > ub_matrix;
    
    % Reflection handling
    reflect_mask_lb = lb_viol & (rand(NP,D) < reflect_prob(:,ones(1,D)));
    reflect_mask_ub = ub_viol & (rand(NP,D) < reflect_prob(:,ones(1,D)));
    
    % Random replacement handling
    rand_mask_lb = lb_viol & ~reflect_mask_lb;
    rand_mask_ub = ub_viol & ~reflect_mask_ub;
    
    % Apply reflections
    offspring(reflect_mask_lb) = 2*lb_matrix(reflect_mask_lb) - offspring(reflect_mask_lb);
    offspring(reflect_mask_ub) = 2*ub_matrix(reflect_mask_ub) - offspring(reflect_mask_ub);
    
    % Apply random replacements
    offspring(rand_mask_lb) = lb_matrix(rand_mask_lb) + rand(sum(rand_mask_lb(:)),1).*(ub_matrix(rand_mask_lb)-lb_matrix(rand_mask_lb));
    offspring(rand_mask_ub) = lb_matrix(rand_mask_ub) + rand(sum(rand_mask_ub(:)),1).*(ub_matrix(rand_mask_ub)-lb_matrix(rand_mask_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
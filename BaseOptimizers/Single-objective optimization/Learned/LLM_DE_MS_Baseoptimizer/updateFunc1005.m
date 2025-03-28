% MATLAB Code
function [offspring] = updateFunc1005(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    norm_cons = tanh(abs(cons));
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Select best solution (feasible first)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask,:);
        x_best = x_best(best_idx,:);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx,:);
    end
    
    % Generate random indices for mutation
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 3));
        r1(i) = ridx(1);
        r2(i) = ridx(2);
        r3(i) = ridx(3);
    end
    
    % Opposition points
    opp_pop = lb + ub - popdecs;
    
    % Adaptive scaling factors
    F = 0.5 + 0.3*(1 - norm_fits) + 0.2*norm_cons;
    
    % Hybrid mutation with opposition learning
    elite_term = bsxfun(@minus, x_best, popdecs);
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    opp_term = popdecs(r3,:) - opp_pop(r3,:);
    mutants = popdecs + F.*elite_term + (1-F).*diff_term + 0.1*norm_cons.*opp_term;
    
    % Constraint-aware crossover
    CR = 0.85 - 0.3*norm_fits + 0.15*norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Intelligent boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Reflection handling
    lb_viol = offspring < lb_matrix;
    ub_viol = offspring > ub_matrix;
    
    reflect_mask_lb = lb_viol & (rand(NP,D) < 0.7);
    reflect_mask_ub = ub_viol & (rand(NP,D) < 0.7);
    
    % Apply reflections
    offspring(reflect_mask_lb) = 2*lb_matrix(reflect_mask_lb) - offspring(reflect_mask_lb);
    offspring(reflect_mask_ub) = 2*ub_matrix(reflect_mask_ub) - offspring(reflect_mask_ub);
    
    % Random replacement for remaining violations
    rand_mask_lb = lb_viol & ~reflect_mask_lb;
    rand_mask_ub = ub_viol & ~reflect_mask_ub;
    
    offspring(rand_mask_lb) = lb_matrix(rand_mask_lb) + rand(sum(rand_mask_lb(:)),1).*(ub_matrix(rand_mask_lb)-lb_matrix(rand_mask_lb));
    offspring(rand_mask_ub) = lb_matrix(rand_mask_ub) + rand(sum(rand_mask_ub(:)),1).*(ub_matrix(rand_mask_ub)-lb_matrix(rand_mask_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
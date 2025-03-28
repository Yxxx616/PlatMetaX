% MATLAB Code
function [offspring] = updateFunc998(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    norm_cons = tanh(abs(cons));
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Select best solution
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Elite direction term
    elite_term = bsxfun(@minus, x_best, popdecs);
    
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
    cons_sum = abs(cons(r1)) + abs(cons(r2));
    weights = exp(-cons_sum);
    weights = weights ./ sum(weights, 2);
    
    % Differential terms with weighted combination
    diff_terms = zeros(NP, D);
    for k = 1:3
        diff_terms = diff_terms + weights(:,k) .* (popdecs(r1(:,k),:) - popdecs(r2(:,k),:));
    end
    
    % Adaptive scaling factors
    F_e = 0.8 * (1 - norm_cons);
    F_d = 0.5 + 0.3 * norm_fits;
    
    % Constraint perturbation
    perturbation = norm_cons .* 0.1 .* randn(NP, 1) .* ones(1, D);
    
    % Mutation
    mutants = popdecs + F_e.*elite_term + F_d.*diff_terms + perturbation;
    
    % Adaptive crossover
    CR = 0.6 + 0.3 * norm_fits;
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
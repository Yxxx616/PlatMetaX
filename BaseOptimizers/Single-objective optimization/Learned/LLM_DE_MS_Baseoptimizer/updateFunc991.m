% MATLAB Code
function [offspring] = updateFunc991(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    norm_fits = (popfits - f_min) / f_range;
    
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    
    % Select top 3 elite solutions based on combined fitness and constraints
    combined = norm_fits + norm_cons;
    [~, elite_idx] = sort(combined);
    elite_pop = popdecs(elite_idx(1:3), :);
    
    % Generate random indices for differential terms
    r1 = zeros(NP, 3);
    r2 = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 6));
        r1(i,:) = ridx(1:3);
        r2(i,:) = ridx(4:6);
    end
    
    % Elite-guided term
    elite_mean = mean(elite_pop, 1);
    elite_term = elite_mean - popdecs;
    
    % Weighted differential term
    weights = exp(-(norm_fits(r1) + norm_fits(r2) + norm_cons(r1) + norm_cons(r2)));
    weights = weights ./ sum(weights, 2);
    diff_terms = sum((popdecs(r1,:) - popdecs(r2,:)) .* weights(:,[1 1 1]), 2);
    
    % Constraint repair term
    repair_term = sign(cons) .* randn(NP, D) .* norm_cons(:, ones(1, D));
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - norm_cons);
    F2 = 0.5 * ones(NP, 1);
    F3 = 0.3 * norm_cons;
    
    % Combined mutation
    mutants = popdecs + F1.*elite_term + F2.*diff_terms + F3.*repair_term;
    
    % Adaptive crossover
    CR = 0.9 * (1 - norm_cons);
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
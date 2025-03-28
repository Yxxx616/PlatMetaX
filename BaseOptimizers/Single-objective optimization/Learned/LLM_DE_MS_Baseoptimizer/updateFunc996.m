% MATLAB Code
function [offspring] = updateFunc996(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints using tanh
    norm_cons = tanh(abs(cons));
    
    % Select top 3 elite solutions based on fitness
    [~, elite_idx] = sort(popfits);
    elite_pop = popdecs(elite_idx(1:3), :);
    elite_mean = mean(elite_pop, 1);
    
    % Elite-guided direction term
    elite_term = bsxfun(@minus, elite_mean, popdecs);
    
    % Generate random indices for differential terms
    r1 = zeros(NP, 3);
    r2 = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        ridx = available(randperm(length(available), 6));
        r1(i,:) = ridx(1:3);
        r2(i,:) = ridx(4:6);
    end
    
    % Fitness-weighted differential term
    weights = exp(-(popfits(r1) + popfits(r2)));
    weights = weights ./ sum(weights, 2);
    diff_terms = sum(bsxfun(@times, popdecs(r1,:) - popdecs(r2,:), ...
        reshape(weights, [NP, 1, 3])), 3);
    
    % Constraint-driven perturbation
    perturbation = norm_cons .* randn(NP, 1) .* ones(1, D);
    
    % Dynamic scaling factors
    F_e = 0.8 * (1 - norm_cons);
    F_d = 0.5 + 0.3 * rand(NP, 1);
    F_p = 0.2 * norm_cons;
    
    % Combined mutation
    mutants = popdecs + F_e.*elite_term + F_d.*diff_terms + F_p.*perturbation;
    
    % Adaptive crossover
    CR = 0.6 + 0.3 * (1 - norm_cons);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
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
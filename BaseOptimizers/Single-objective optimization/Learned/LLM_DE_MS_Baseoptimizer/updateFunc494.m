% MATLAB Code
function [offspring] = updateFunc494(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate population statistics
    pop_mean = mean(popdecs);
    pop_std = std(popdecs);
    max_std = max(pop_std);
    
    % Identify elite solution considering both fitness and constraints
    penalty = 1e6 * max(0, cons);
    combined = popfits + penalty;
    [~, elite_idx] = min(combined);
    elite = popdecs(elite_idx, :);
    
    % Calculate fitness-based weights using exponential ranking
    [~, rank_order] = sort(popfits);
    ranks = zeros(size(popfits));
    ranks(rank_order) = 1:NP;
    w_fit = exp(-5 * ranks/NP)';
    
    % Calculate constraint-based weights
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    w_con = 1 - exp(-5 * abs_cons/(max_con + 1e-12));
    
    % Adaptive scaling factors
    diversity = mean(pop_std)/(max_std + 1e-12);
    F1 = 0.5 + 0.3 * tanh(diversity);
    F2 = 0.3 * (1 - F1);
    F3 = 0.2 * (1 - F1);
    F4 = 0.1 * rand();
    
    % Generate random indices (vectorized)
    [~, idx] = sort(rand(NP, NP), 2);
    mask = idx ~= (1:NP)';
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1);
    for i = 1:NP
        candidates = find(mask(i,:));
        r1(i) = candidates(1);
        r2(i) = candidates(2);
        r3(i) = candidates(3);
    end
    
    % Mutation components
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_fit = popdecs(r1,:) - popdecs(r2,:);
    diff_con = popdecs(r2,:) - popdecs(r3,:);
    
    % Apply weights
    diff_fit = diff_fit .* repmat(w_fit, 1, D);
    diff_con = diff_con .* repmat(w_con, 1, D);
    
    % Generate offspring
    offspring = popdecs + F1.*elite_dir + F2.*diff_fit + F3.*diff_con + ...
                F4.*(ub-lb).*randn(NP,D);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring = offspring.*(~below & ~above) + ...
                (2*lb_rep - offspring).*below + ...
                (2*ub_rep - offspring).*above;
    
    % Final bounds check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
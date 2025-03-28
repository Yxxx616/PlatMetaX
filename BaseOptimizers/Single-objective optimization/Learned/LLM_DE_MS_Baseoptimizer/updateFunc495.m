% MATLAB Code
function [offspring] = updateFunc495(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Current iteration estimation (normalized 0-1)
    t = 0.5; % Placeholder for actual iteration tracking
    T = 1;   % Max iterations placeholder
    
    % Identify elite solution considering constraints
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
    F1 = 0.6 * (1 - t/T);
    F2 = 0.4 * t/T;
    F3 = 0.2 * (1 + sum(abs_cons)/(NP * (max_con + 1e-12));
    F4 = 0.1 * (1 - t/T);
    sigma = 0.2 * (ub - lb) * (1 - t/T);
    
    % Generate random indices (vectorized)
    [~, idx] = sort(rand(NP, NP), 2);
    mask = idx ~= (1:NP)';
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = find(mask(i,:));
        r1(i) = candidates(1);
        r2(i) = candidates(2);
        r3(i) = candidates(3);
        r4(i) = candidates(4);
    end
    
    % Mutation components
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_fit = popdecs(r1,:) - popdecs(r2,:);
    diff_con = popdecs(r3,:) - popdecs(r4,:);
    rand_pert = sigma .* randn(NP,D);
    
    % Apply weights and scaling
    xi = rand(NP,1);
    diff_fit = diff_fit .* repmat(w_fit, 1, D);
    diff_con = diff_con .* repmat(w_con, 1, D) .* repmat(xi, 1, D);
    
    % Generate offspring
    offspring = popdecs + F1.*elite_dir + F2.*diff_fit + F3.*diff_con + F4.*rand_pert;
    
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
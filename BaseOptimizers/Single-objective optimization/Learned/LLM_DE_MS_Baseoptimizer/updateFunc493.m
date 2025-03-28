% MATLAB Code
function [offspring] = updateFunc493(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution with combined fitness and constraint info
    combined_scores = popfits + 1e6*max(0, cons); % Penalize infeasible solutions
    [~, elite_idx] = min(combined_scores);
    elite = popdecs(elite_idx, :);
    
    % Calculate adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_fit = 1 - (popfits - min_fit) / (max_fit - min_fit + eps);
    
    max_con = max(abs(cons));
    w_con = abs(cons) / (max_con + eps);
    
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
    
    % Adaptive scaling factors based on population diversity
    mean_dist = mean(pdist(popdecs));
    max_dist = max(pdist(popdecs));
    F1 = 0.7 * (1 - exp(-mean_dist/max_dist));
    F2 = 0.5 * (1 - F1);
    F3 = 0.3 * (1 - F1);
    F4 = 0.1 + 0.1*rand();
    
    % Mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_fit = popdecs(r1,:) - popdecs(r2,:);
    diff_con = popdecs(r2,:) - popdecs(r3,:);
    
    % Apply adaptive weights
    diff_fit = diff_fit .* repmat(w_fit, 1, D);
    diff_con = diff_con .* repmat(w_con, 1, D);
    
    % Generate offspring with adaptive components
    offspring = popdecs + F1.*elite_dir + F2.*diff_fit + F3.*diff_con + ...
                F4*(ub-lb).*randn(NP,D);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflect back into bounds
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring = offspring.*(~below & ~above) + ...
                (2*lb_rep - offspring).*below + ...
                (2*ub_rep - offspring).*above;
    
    % Final bounds check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
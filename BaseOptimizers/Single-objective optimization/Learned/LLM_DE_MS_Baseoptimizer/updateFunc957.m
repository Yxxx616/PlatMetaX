% MATLAB Code
function [offspring] = updateFunc957(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute feasibility-weighted centroid
    f_mean = mean(popfits);
    f_min = min(popfits);
    f_max = max(popfits);
    phi_max = max(abs(cons));
    
    % Weight calculation
    weights = 1./(1 + max(0, cons) + abs(popfits - f_mean));
    weights = weights / sum(weights); % Normalize
    centroid = weights' * popdecs;
    
    % 2. Identify best and worst solutions
    [~, best_idx] = min(popfits + 10*max(0, cons));
    [~, worst_idx] = max(popfits + 10*max(0, cons));
    x_best = popdecs(best_idx,:);
    x_worst = popdecs(worst_idx,:);
    
    % 3. Compute adaptive F and CR
    F = 0.4 + 0.5 * (1 - abs(cons)/(phi_max + eps)) .* (1 - (popfits - f_min)/(f_max - f_min + eps));
    CR = 0.9 * (1 - abs(cons)/(phi_max + eps)).^0.5;
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Enhanced directional mutation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    best_worst_diff = x_best - x_worst;
    
    mutants = centroid(ones(NP,1),:) + F.*diff1 + (1-F).*diff2 + 0.1*best_worst_diff(ones(NP,1),:);
    
    % 6. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with adaptive reflection
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_matrix;
    above_ub = offspring > ub_matrix;
    
    % Adaptive reflection based on fitness rank
    [~, rank_order] = sort(popfits + 10*max(0, cons));
    rank_coeff = (rank_order/NP).^2;
    
    offspring(below_lb) = lb_matrix(below_lb) + rank_coeff(below_lb(:,1)).*(popdecs(below_lb) - lb_matrix(below_lb));
    offspring(above_ub) = ub_matrix(above_ub) - rank_coeff(above_ub(:,1)).*(ub_matrix(above_ub) - popdecs(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
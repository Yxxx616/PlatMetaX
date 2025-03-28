% MATLAB Code
function [offspring] = updateFunc1196(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate weights
    feasible_mask = cons <= 0;
    n_feas = sum(feasible_mask);
    
    % Feasibility weight
    if n_feas > 0
        mean_cons = mean(abs(cons(feasible_mask)));
        w_feas = (n_feas/NP) * exp(-mean_cons/(max(abs(cons))+eps));
    else
        w_feas = 0.1;
    end
    
    % Fitness weights (normalized)
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_fit = exp(-(popfits - min_fit)/(max_fit - min_fit + eps));
    w_fit = w_fit / sum(w_fit);
    
    % Diversity weights
    centroid = mean(popdecs, 1);
    dist = sqrt(sum((popdecs - centroid).^2, 2));
    [~, div_rank] = sort(dist, 'descend');
    w_div = 1 - (div_rank-1)/NP;
    
    % 2. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    if n_feas > 0
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    [~, div_idx] = max(dist);
    x_div = popdecs(div_idx, :);
    
    % 3. Generate distinct random indices
    idx = (1:NP)';
    R = zeros(NP, 6);
    for i = 1:6
        R(:,i) = randperm(NP)';
        while any(R(:,i) == idx)
            R(:,i) = randperm(NP)';
        end
    end
    
    % 4. Mutation with adaptive F
    F1 = 0.8 * (1 + 0.1 * randn(NP, 1));
    F2 = 0.7 * (1 + 0.2 * randn(NP, 1));
    F3 = 0.6 * (1 + 0.3 * randn(NP, 1));
    
    v_feas = x_feas + F1 .* (popdecs(R(:,1),:) - popdecs(R(:,2),:);
    v_best = x_best + F2 .* (popdecs(R(:,3),:) - popdecs(R(:,4),:);
    v_div = x_div + F3 .* (popdecs(R(:,5),:) - popdecs(R(:,6),:);
    
    % 5. Weighted combination
    weights = [w_feas*ones(NP,1), w_fit, w_div];
    weights = weights ./ sum(weights, 2);
    mutants = weights(:,1).*v_feas + weights(:,2).*v_best + weights(:,3).*v_div;
    
    % 6. Adaptive crossover
    CR = 0.5 + 0.4*(popfits - min_fit)/(max_fit - min_fit + eps);
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc217(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    sum_con = sum(abs_cons) + eps;
    
    % Fitness statistics
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    
    % Population diversity
    diversity = mean(std(popdecs)) + eps;
    
    % Identify key individuals
    [~, elite_idx] = min(popfits + 1e6*abs_cons);
    elite = popdecs(elite_idx, :);
    
    feasible = cons <= 0;
    if any(feasible)
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible,:);
        best_feas = best_feas(best_feas_idx,:);
    else
        [~, min_con_idx] = min(abs_cons);
        best_feas = popdecs(min_con_idx,:);
    end
    
    % Calculate adaptive mutation factors
    F1 = 0.7 * (1 - abs_cons/max_con);
    F2 = 0.5 * tanh((popfits-avg_fit)/std_fit) .* (1 - abs_cons/sum_con);
    F3 = 0.3 * diversity/D;
    F4 = 0.1 * abs_cons/max_con;
    
    % Generate mutation components
    v1 = popdecs + bsxfun(@times, F1, elite - popdecs);
    v2 = popdecs + bsxfun(@times, F2, best_feas - popdecs);
    v3 = popdecs + bsxfun(@times, F3, diversity * randn(NP,D));
    v4 = popdecs + bsxfun(@times, F4, lb + ub - 2*popdecs + randn(NP,D));
    
    % Adaptive combination weights based on component performance
    comp_perf = [F1, F2, F3, F4];
    [~, rank_order] = sort(comp_perf, 2, 'descend');
    rank_weights = exp(-[0.5, 0.3, 0.2, 0.1] .* (rank_order-1));
    norm_weights = rank_weights ./ sum(rank_weights, 2);
    
    % Combine components with adaptive weights
    offspring = bsxfun(@times, norm_weights(:,1), v1) + ...
                bsxfun(@times, norm_weights(:,2), v2) + ...
                bsxfun(@times, norm_weights(:,3), v3) + ...
                bsxfun(@times, norm_weights(:,4), v4);
    
    % Boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.1 + 0.4*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final adaptive perturbation
    perturb_scale = 0.01 * (1 + abs_cons/max_con);
    offspring = offspring + bsxfun(@times, perturb_scale, diversity.*randn(NP,D));
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub), lb);
end
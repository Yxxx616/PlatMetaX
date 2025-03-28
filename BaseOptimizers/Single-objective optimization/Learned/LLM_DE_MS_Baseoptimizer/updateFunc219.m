% MATLAB Code
function [offspring] = updateFunc219(popdecs, popfits, cons)
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
    norm_fits = (popfits - avg_fit) / std_fit;
    
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
    F2 = 0.5 * tanh(norm_fits) .* (1 - abs_cons/sum_con);
    F3 = 0.3 * diversity/D * ones(NP,1);
    F4 = 0.1 * abs_cons/max_con;
    
    % Generate mutation components
    v1 = popdecs + bsxfun(@times, F1, elite - popdecs);
    v2 = popdecs + bsxfun(@times, F2, best_feas - popdecs);
    v3 = popdecs + bsxfun(@times, F3, diversity * randn(NP,D));
    v4 = popdecs + bsxfun(@times, F4, lb + ub - 2*popdecs + randn(NP,D));
    
    % Rank-based adaptive combination
    F_matrix = [F1, F2, F3, F4];
    [~, ranks] = sort(F_matrix, 2);
    weights = exp(-0.5 * (ranks-1));
    norm_weights = weights ./ sum(weights, 2);
    
    % Combine components
    offspring = bsxfun(@times, norm_weights(:,1), v1) + ...
                bsxfun(@times, norm_weights(:,2), v2) + ...
                bsxfun(@times, norm_weights(:,3), v3) + ...
                bsxfun(@times, norm_weights(:,4), v4);
    
    % Adaptive boundary handling
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.25 + 0.25*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final adaptive perturbation
    perturb_scale = 0.01 * (1 + abs_cons/max_con);
    offspring = offspring + bsxfun(@times, perturb_scale, diversity.*randn(NP,D));
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub), lb);
end
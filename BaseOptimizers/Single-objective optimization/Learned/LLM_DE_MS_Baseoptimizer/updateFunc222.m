% MATLAB Code
function [offspring] = updateFunc222(popdecs, popfits, cons)
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
    min_fit = min(popfits);
    max_fit = max(popfits);
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
    F1 = 0.8 * (1 - abs_cons/max_con) .* tanh(norm_fits);
    F2 = 0.6 ./ (1 + exp(abs_cons/sum_con));
    F3 = 0.4 * diversity/D .* (1 - (popfits-min_fit)/(max_fit-min_fit+eps));
    F4 = 0.2 * abs_cons/max_con;
    
    % Generate mutation components (vectorized)
    v1 = popdecs + bsxfun(@times, F1, elite - popdecs);
    v2 = popdecs + bsxfun(@times, F2, best_feas - popdecs);
    v3 = popdecs + bsxfun(@times, F3, diversity * randn(NP,D));
    v4 = popdecs + bsxfun(@times, F4, lb + ub - 2*popdecs + randn(NP,D));
    
    % Constraint-based adaptive combination
    [~, con_ranks] = sort(abs_cons);
    weights = zeros(NP,4);
    for i = 1:NP
        r = con_ranks(i)/NP;
        if r < 0.25
            weights(i,:) = [0.5, 0.3, 0.15, 0.05];
        elseif r < 0.5
            weights(i,:) = [0.4, 0.4, 0.15, 0.05];
        elseif r < 0.75
            weights(i,:) = [0.3, 0.3, 0.3, 0.1];
        else
            weights(i,:) = [0.2, 0.2, 0.3, 0.3];
        end
    end
    
    % Combine components with vectorized operations
    offspring = bsxfun(@times, weights(:,1), v1) + ...
                bsxfun(@times, weights(:,2), v2) + ...
                bsxfun(@times, weights(:,3), v3) + ...
                bsxfun(@times, weights(:,4), v4);
    
    % Adaptive boundary handling
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.1 + 0.2*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(lb - offspring)).*out_low + ...
               (ub - reflect_coeff.*(offspring - ub)).*out_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc223(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    min_con = min(abs_cons);
    
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
    
    % Constraint-adaptive weights
    weights = 1 ./ (1 + exp(5 * (abs_cons - min_con) / (max_con - min_con + eps));
    
    % Calculate adaptive mutation factors
    F1 = 0.9 * tanh(norm_fits);
    F2 = 0.5 * (1 - abs_cons/max_con);
    F3 = 0.3 * diversity/D;
    F4 = 0.1 * abs_cons/max_con;
    
    % Generate mutation components (vectorized)
    v1 = popdecs + bsxfun(@times, F1, elite - popdecs) + bsxfun(@times, F2, best_feas - popdecs);
    v2 = popdecs + bsxfun(@times, F3.*(1-weights), diversity * randn(NP,D));
    v3 = popdecs + bsxfun(@times, F4.*weights, lb + ub - 2*popdecs);
    
    % Hybrid combination
    offspring = bsxfun(@times, weights, v1) + ...
                bsxfun(@times, (1-weights), 0.7*v2 + 0.3*v3);
    
    % Adaptive boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.2 * rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(lb - offspring)).*out_low + ...
               (ub - reflect_coeff.*(offspring - ub)).*out_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
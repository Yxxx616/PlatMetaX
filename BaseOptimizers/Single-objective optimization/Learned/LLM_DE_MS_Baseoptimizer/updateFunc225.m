% MATLAB Code
function [offspring] = updateFunc225(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    min_con = min(abs_cons);
    
    % Calculate weights based on constraint violations
    weights = 1 ./ (1 + exp(5 * (abs_cons - min_con) ./ (max_con - min_con + eps)));
    
    % Fitness statistics
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - avg_fit) ./ std_fit;
    
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
    
    % Calculate adaptive scaling factors
    F1 = 0.5 * (1 + tanh(norm_fits));
    F2 = 0.3 * (1 - abs_cons/max_con);
    F3 = 0.2 * diversity/D;
    
    % Generate directional vectors
    d_elite = bsxfun(@minus, elite, popdecs);
    d_feas = bsxfun(@minus, best_feas, popdecs);
    d_div = diversity * randn(NP,D);
    
    % Combine mutation components
    weighted_part = bsxfun(@times, F1, d_elite) + bsxfun(@times, F2, d_feas);
    random_part = bsxfun(@times, F3, d_div);
    
    offspring = popdecs + bsxfun(@times, weights, weighted_part) + ...
                bsxfun(@times, (1-weights), random_part);
    
    % Boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.5 * rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(lb - offspring)).*out_low + ...
               (ub - reflect_coeff.*(offspring - ub)).*out_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end
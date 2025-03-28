% MATLAB Code
function [offspring] = updateFunc227(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints with improved normalization
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    min_con = min(abs_cons);
    weights = 1 ./ (1 + exp(5 * (abs_cons - min_con) ./ (max_con - min_con + eps));
    
    % Enhanced fitness normalization
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - avg_fit) ./ std_fit;
    
    % Improved diversity measure
    diversity = mean(std(popdecs)) + eps;
    
    % Elite selection with constraint consideration
    [~, elite_idx] = min(popfits + 1e6*abs_cons);
    elite = popdecs(elite_idx, :);
    
    % Best feasible solution selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible,:);
        best_feas = best_feas(best_feas_idx,:);
    else
        [~, min_con_idx] = min(abs_cons);
        best_feas = popdecs(min_con_idx,:);
    end
    
    % Adaptive scaling factors with improved balance
    F1 = 0.7 * (1 - tanh(norm_fits));
    F2 = 0.2 * (1 - abs_cons/max_con);
    F3 = 0.1 * diversity/D;
    
    % Vectorized direction calculations
    d_elite = bsxfun(@minus, elite, popdecs);
    d_feas = bsxfun(@minus, best_feas, popdecs);
    d_div = diversity * randn(NP,D);
    
    % Combined mutation with improved weighting
    weighted_part = bsxfun(@times, F1, d_elite) + bsxfun(@times, F2, d_feas);
    random_part = bsxfun(@times, F3, d_div);
    
    offspring = popdecs + bsxfun(@times, weights, weighted_part) + ...
                bsxfun(@times, (1-weights), random_part);
    
    % Enhanced boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.5 * rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(lb - offspring)).*out_low + ...
               (ub - reflect_coeff.*(offspring - ub)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end
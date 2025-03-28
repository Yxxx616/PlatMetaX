% MATLAB Code
function [offspring] = updateFunc215(popdecs, popfits, cons)
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
    F1 = 0.8 * (1 - abs_cons/max_con);
    F2 = 0.6 * tanh((popfits-avg_fit)/std_fit) .* (abs_cons/sum_con);
    F3 = 0.4 * diversity/D;
    F4 = 0.2 * abs_cons/max_con;
    
    % Generate mutation components
    elite_diff = bsxfun(@times, F1, elite - popdecs);
    feas_repair = bsxfun(@times, F2, best_feas - popdecs);
    rand_perturb = bsxfun(@times, F3, diversity * randn(NP,D));
    opposition = bsxfun(@times, F4, lb + ub - popdecs + randn(NP,D));
    
    % Adaptive combination weights
    w1 = 0.5 + 0.3*rand(NP,1);
    w2 = 0.3 + 0.4*rand(NP,1);
    w3 = 0.1 + 0.2*rand(NP,1);
    w4 = 0.1 + 0.1*rand(NP,1);
    
    % Combine components
    offspring = popdecs + ...
        bsxfun(@times, w1, elite_diff) + ...
        bsxfun(@times, w2, feas_repair) + ...
        bsxfun(@times, w3, rand_perturb) + ...
        bsxfun(@times, w4, opposition);
    
    % Smart boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.3 + 0.5*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final adaptive perturbation
    perturb_scale = 0.01 * (1 + abs_cons/max_con);
    offspring = offspring + bsxfun(@times, perturb_scale, diversity.*randn(NP,D));
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc211(popdecs, popfits, cons)
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
    
    % Calculate mutation factors (improved version)
    F1 = 0.7 * (1 - abs_cons/max_con);
    F2 = 0.3 * tanh((popfits-avg_fit)/std_fit) .* (abs_cons/sum_con);
    F3 = 0.2 * diversity/D;
    F4 = 0.1 * abs_cons/max_con;
    
    % Generate mutation components
    elite_diff = elite - popdecs;
    feas_repair = best_feas - popdecs;
    rand_perturb = diversity * randn(NP,D);
    opposition = lb + ub - popdecs - randn(NP,D);
    
    % Adaptive scaling factor for constraint-aware component
    xi = 0.7 + 0.6*rand(NP,1);
    
    % Combine components (improved weighting)
    offspring = popdecs + ...
        F1.*elite_diff + ...
        xi.*F2.*feas_repair + ...
        F3.*rand_perturb.*(1 + abs_cons/max_con) + ...
        F4.*opposition;
    
    % Enhanced boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.2 + 0.5*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Adaptive final perturbation
    perturb_scale = 0.03 * (1 + abs_cons/max_con);
    offspring = offspring + perturb_scale.*diversity.*randn(NP,D);
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub), lb);
end
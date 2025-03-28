% MATLAB Code
function [offspring] = updateFunc208(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate constraint statistics
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    sum_con = sum(abs_cons) + eps;
    
    % Calculate fitness statistics
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    
    % Calculate population diversity
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
    
    [~, worst_infeas_idx] = max(abs_cons);
    worst_infeas = popdecs(worst_infeas_idx,:);
    
    % Vectorized mutation factors
    F1 = 0.4 * (1 - abs_cons/max_con);
    F2 = 0.3 * tanh((popfits-avg_fit)/std_fit);
    F3 = 0.2 * abs_cons/max_con;
    F4 = 0.1 * diversity/D;
    
    % Generate all perturbations at once
    elite_diff = elite - popdecs;
    feas_repair = (best_feas - popdecs) .* (abs_cons/sum_con);
    rand_perturb = diversity * randn(NP,D);
    con_repel = (popdecs - worst_infeas) .* (cons/max_con);
    
    % Combine components
    offspring = popdecs + ...
        F1.*elite_diff + ...
        F2.*feas_repair + ...
        F3.*con_repel + ...
        F4.*rand_perturb;
    
    % Adaptive boundary handling
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.2 + 0.3*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final small perturbation scaled by constraint violation
    perturb_scale = 0.05 * (1 + abs_cons/max_con);
    offspring = offspring + perturb_scale.*diversity.*randn(NP,D);
    offspring = max(min(offspring, ub), lb);
end
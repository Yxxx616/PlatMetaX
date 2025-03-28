% MATLAB Code
function [offspring] = updateFunc207(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite individual (minimize penalized fitness)
    penalty = popfits + 1e6*abs(cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    % Identify best feasible and worst infeasible
    feasible = cons <= 0;
    if any(feasible)
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible,:);
        best_feas = best_feas(best_feas_idx,:);
    else
        [~, min_con_idx] = min(abs(cons));
        best_feas = popdecs(min_con_idx,:);
    end
    
    [~, worst_infeas_idx] = max(abs(cons));
    worst_infeas = popdecs(worst_infeas_idx,:);
    
    % Calculate statistics
    max_con = max(abs(cons)) + eps;
    sum_con = sum(abs(cons)) + eps;
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    diversity = mean(std(popdecs)) + eps;
    
    % Generate scaling factors
    F1 = 0.5 * (1 - abs(cons)/max_con);
    F2 = 0.3 * tanh((popfits-avg_fit)/std_fit);
    F3 = 0.1 * diversity/D;
    F4 = 0.2 * abs(cons)/max_con;
    
    % Vectorized mutation
    for i = 1:NP
        elite_diff = elite - popdecs(i,:);
        feas_repair = (best_feas - popdecs(i,:)) .* (abs(cons(i))/sum_con);
        rand_perturb = diversity * randn(1,D);
        con_perturb = (worst_infeas - popdecs(i,:)) .* (cons(i)/max_con);
        
        offspring(i,:) = popdecs(i,:) + ...
            F1(i) * elite_diff + ...
            F2(i) * feas_repair + ...
            F3 * rand_perturb + ...
            F4(i) * con_perturb;
    end
    
    % Adaptive boundary handling
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    
    % Reflection with adaptive coefficient
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.2 + 0.3*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final small perturbation scaled by constraint violation
    perturb_scale = 0.05 * (1 + abs(cons)/max_con);
    offspring = offspring + perturb_scale.*diversity.*randn(NP,D);
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc205(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite individual (minimize fitness while satisfying constraints)
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
        [~, best_feas_idx] = min(abs(cons));
        best_feas = popdecs(best_feas_idx,:);
    end
    
    [~, worst_infeas_idx] = max(abs(cons));
    worst_infeas = popdecs(worst_infeas_idx,:);
    
    % Calculate statistics
    max_con = max(abs(cons)) + eps;
    sum_con = sum(abs(cons)) + eps;
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    
    % Generate random indices for differential components
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    
    % Vectorized mutation
    for i = 1:NP
        % Adaptive scaling factors
        F1 = 0.8 * (1 - abs(cons(i))/max_con);
        F2 = 0.5 ./ (1 + exp(-(popfits(i)-avg_fit)/std_fit));
        F3 = 0.3 * abs(cons(i))/sum_con;
        F4 = 0.2 * tanh(abs(cons(i)));
        
        % Mutation components
        elite_diff = elite - popdecs(i,:);
        rank_diff1 = popdecs(r1(i),:) - popdecs(r2(i),:);
        rank_diff2 = popdecs(r3(i),:) - popdecs(r4(i),:);
        cons_repair = (best_feas - worst_infeas) .* randn(1,D);
        
        % Combined mutation
        offspring(i,:) = popdecs(i,:) + ...
            F1 * elite_diff + ...
            F2 * rank_diff1 + ...
            F3 * rank_diff2 + ...
            F4 * cons_repair;
    end
    
    % Boundary handling with adaptive reflection
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    
    % Reflection with adaptive coefficient
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.2 + 0.6*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final small perturbation (scaled by population diversity)
    diversity = mean(std(popdecs)) + eps;
    offspring = offspring + 0.05*diversity*randn(NP,D);
    offspring = max(min(offspring, ub), lb);
end
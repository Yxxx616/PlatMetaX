% MATLAB Code
function [offspring] = updateFunc206(popdecs, popfits, cons)
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
    diversity = mean(std(popdecs)) + eps;
    
    % Rank-based weights
    [~, rank_idx] = sort(popfits);
    rank_weights = (NP:-1:1)'/NP;
    
    % Generate random indices for differential components
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    % Vectorized mutation
    for i = 1:NP
        % Adaptive scaling factors
        F1 = 0.8 * (1 - abs(cons(i))/max_con);
        F2 = 0.5 * tanh((popfits(i)-avg_fit)/std_fit);
        F3 = 0.3 * abs(cons(i))/max_con;
        F4 = 0.2 * rank_weights(rank_idx == i);
        
        % Mutation components
        elite_diff = elite - popdecs(i,:);
        feas_diff = best_feas - worst_infeas;
        random_perturb = diversity * randn(1,D);
        repair_vec = (best_feas - popdecs(i,:)) .* (abs(cons(i))/sum_con);
        
        % Combined mutation
        offspring(i,:) = popdecs(i,:) + ...
            F1 * elite_diff + ...
            F2 * feas_diff .* randn(1,D) + ...
            F3 * random_perturb + ...
            F4 * repair_vec;
    end
    
    % Boundary handling with adaptive reflection
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    
    % Reflection with adaptive coefficient
    out_low = offspring < lb;
    out_high = offspring > ub;
    reflect_coeff = 0.1 + 0.4*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + reflect_coeff.*(popdecs - lb)).*out_low + ...
               (ub - reflect_coeff.*(ub - popdecs)).*out_high;
    
    % Final small perturbation (scaled by constraint violation)
    perturb_scale = 0.05 * (1 + abs(cons)/max_con);
    offspring = offspring + perturb_scale.*diversity.*randn(NP,D);
    offspring = max(min(offspring, ub), lb);
end
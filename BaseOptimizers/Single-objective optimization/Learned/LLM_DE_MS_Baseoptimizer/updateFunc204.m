% MATLAB Code
function [offspring] = updateFunc204(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*abs(cons));
    elite = popdecs(elite_idx, :);
    
    % Sort population by constrained fitness
    [~, sorted_idx] = sort(popfits + 1e6*abs(cons));
    top_rank = popdecs(sorted_idx(1:ceil(NP/3)), :);
    mid_rank = popdecs(sorted_idx(ceil(NP/3)+1:floor(2*NP/3))), :);
    bot_rank = popdecs(sorted_idx(floor(2*NP/3)+1:end), :);
    
    % Find best feasible and worst infeasible
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        [~, best_feas_idx] = min(abs(cons));
        best_feas = popdecs(best_feas_idx, :);
    end
    
    [~, worst_infeas_idx] = max(abs(cons));
    worst_infeas = popdecs(worst_infeas_idx, :);
    
    % Calculate statistics
    sum_con = sum(abs(cons)) + eps;
    max_con = max(abs(cons)) + eps;
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    
    % Generate random indices for differential components
    r1 = randi(size(top_rank, 1), NP, 1);
    r2 = randi(size(mid_rank, 1), NP, 1);
    r3 = randi(size(bot_rank, 1), NP, 1);
    s1 = randi(size(top_rank, 1), NP, 1);
    s2 = randi(size(mid_rank, 1), NP, 1);
    s3 = randi(size(bot_rank, 1), NP, 1);
    
    % Vectorized mutation
    for i = 1:NP
        % Adaptive scaling factors
        F1 = 0.9 * (1 - abs(cons(i))/max_con);
        F2 = 0.5 * tanh((popfits(i) - avg_fit)/std_fit);
        F3 = 0.3 * abs(cons(i))/sum_con;
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        
        % Weighted rank differences
        rank_diff = 0.6*(top_rank(r1(i), :) - top_rank(s1(i), :)) + ...
                   0.3*(mid_rank(r2(i), :) - mid_rank(s2(i), :)) + ...
                   0.1*(bot_rank(r3(i), :) - bot_rank(s3(i), :));
        
        % Constraint-driven perturbation
        cons_pert = (best_feas - worst_infeas) .* randn(1, D) * F3;
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + ...
            F1 * elite_diff + ...
            F2 * rank_diff + ...
            cons_pert;
    end
    
    % Boundary handling with adaptive reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    out_of_bounds = (offspring < lb) | (offspring > ub);
    if any(out_of_bounds(:))
        beta = 0.1 + 0.8*rand(NP, D);
        reflection_low = lb + beta .* (popdecs - lb);
        reflection_high = ub - beta .* (ub - popdecs);
        offspring = offspring .* ~out_of_bounds + ...
                   reflection_low .* (offspring < lb) + ...
                   reflection_high .* (offspring > ub);
    end
    
    % Final adaptive perturbation
    perturbation_scale = 0.1 * (max(popfits) - min(popfits))/(std_fit + eps);
    offspring = offspring + perturbation_scale * randn(NP, D);
    offspring = max(min(offspring, ub), lb);
end
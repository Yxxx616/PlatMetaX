% MATLAB Code
function [offspring] = updateFunc1553(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-6;
    
    % 1. Calculate constraint-aware weights
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons);
    w_feas = 1 ./ (1 + exp(-5*(abs_cons - mean_cons)./max(std_cons, eps));
    
    % 2. Select elite solutions (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.3 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    
    % 3. Calculate feasible direction
    feasible = cons <= 0;
    if any(feasible)
        mean_feas = mean(popdecs(feasible,:), 1);
    else
        mean_feas = mean(popdecs, 1);
    end
    feas_dir = mean_feas - popdecs;
    
    % 4. Elite direction
    elite_dir = elite - popdecs;
    
    % 5. Hybrid mutation with adaptive scaling
    F = 0.5 + 0.3 * (1 - w_feas);
    mutants = popdecs + F .* (w_feas.*feas_dir + (1-w_feas).*elite_dir);
    
    % 6. Adaptive crossover
    CR = 0.2 + 0.6 * w_feas;
    mask_cr = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 7. Boundary handling with midpoint reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = (lb(mask_lb) + popdecs(mask_lb)) / 2;
    offspring(mask_ub) = (ub(mask_ub) + popdecs(mask_ub)) / 2;
    
    % 8. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 9. Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb) .* (1 - w_feas(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
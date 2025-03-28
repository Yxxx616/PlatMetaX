% MATLAB Code
function [offspring] = updateFunc1550(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Calculate feasibility weights
    abs_cons = abs(cons);
    mean_cons = mean(abs_cons);
    std_cons = std(abs_cons);
    w_feas = 1 ./ (1 + exp(-5*(abs_cons - mean_cons)./max(std_cons, 1e-6));
    
    % 2. Select elite solutions (top 20%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.2 * NP));
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
    
    % 4. Opposition-based direction
    opposite_dir = (lb + ub - popdecs) - popdecs;
    
    % 5. Elite direction
    elite_dir = elite - popdecs;
    
    % 6. Adaptive weights
    w_elite = 0.5 * (1 - w_feas);
    w_opp = 0.5 * (1 - w_feas);
    
    % 7. Generate mutation vectors
    mutants = popdecs + w_elite.*elite_dir + w_feas.*feas_dir + w_opp.*opposite_dir;
    
    % 8. Dynamic crossover
    CR = 0.1 + 0.7 * w_feas;
    mask_cr = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 9. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = lb(mask_lb) + 0.5 * rand(sum(sum(mask_lb)),1) .* (popdecs(mask_lb) - lb(mask_lb));
    offspring(mask_ub) = ub(mask_ub) - 0.5 * rand(sum(sum(mask_ub)),1) .* (ub(mask_ub) - popdecs(mask_ub));
    
    % 10. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 11. Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb) .* (1 - w_feas(top_idx));
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
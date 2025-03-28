% MATLAB Code
function [offspring] = updateFunc352(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint handling
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    
    % Elite selection (best overall fitness)
    [~, elite_idx] = min(popfits);
    elite = popdecs(elite_idx,:);
    
    % Best feasible individual
    if any(feasible_mask)
        feasible_fits = popfits(feasible_mask);
        [~, best_feas_idx] = min(feasible_fits);
        temp = find(feasible_mask);
        best_feas = popdecs(temp(best_feas_idx),:);
    else
        [~, best_cons_idx] = min(abs_cons);
        best_feas = popdecs(best_cons_idx,:);
    end
    
    % Generate unique random pairs avoiding current index
    r1 = arrayfun(@(x) randi([1, NP-1]), 1:NP)';
    r1 = r1 + (r1 >= (1:NP)');
    r2 = arrayfun(@(x) randi([1, NP-1]), 1:NP)';
    r2 = r2 + (r2 >= (1:NP)');
    
    % Adaptive parameters
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_std = std(abs_cons) + eps;
    
    % Adaptive weights
    alpha = 1./(1 + exp(-(popfits - f_mean)./f_std));
    beta = 1./(1 + exp(abs_cons./c_std));
    
    % Rank-based scaling factors
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    gamma = 0.5 + 0.3 * (1 - (ranks/NP).^0.5);
    
    % Direction vectors
    d_feas = best_feas - popdecs;
    d_elite = elite - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation with adaptive weights
    base_mutation = beta.*d_feas + (1-beta).*(alpha.*d_elite + (1-alpha).*d_rand);
    offspring = popdecs + gamma.*base_mutation + ...
                0.1*(1-gamma).*randn(NP,D).*(ub-lb);
    
    % Boundary handling with randomized reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    rand_scale = rand(NP,D);
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + abs(offspring - lb).*rand_scale).*out_low + ...
               (ub - abs(offspring - ub).*rand_scale).*out_high;
end
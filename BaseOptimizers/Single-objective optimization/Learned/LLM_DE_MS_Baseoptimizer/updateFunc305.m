% MATLAB Code
function [offspring] = updateFunc305(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = max(0, cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons) + eps;
    norm_cons = (abs_cons - c_min) / (c_max - c_min + eps);
    
    f_min = min(popfits);
    f_max = max(popfits) + eps;
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    % Identify key individuals
    feasible_mask = cons <= 0;
    
    % Elite individual (best feasible or least infeasible)
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(abs_cons);
        elite = popdecs(elite_idx,:);
    end
    
    % Best feasible individual
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask,:);
        best_feas = best_feas(best_idx,:);
    else
        best_feas = elite;
    end
    
    % Individual with lowest constraint violation
    [~, lowcons_idx] = min(abs_cons);
    lowcons = popdecs(lowcons_idx,:);
    
    % Generate unique random pairs
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(same_idx)
        r1(same_idx) = randi(NP, sum(same_idx), 1);
        r2(same_idx) = randi(NP, sum(same_idx), 1);
        same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Adaptive parameter
    sigma = 0.2 + 0.3*rand();
    
    % Calculate direction vectors
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    d_cons = repmat(lowcons, NP, 1) - popdecs;
    
    % Calculate weights using modified softmax
    w_elite = exp(-norm_cons/sigma);
    w_feas = exp(-norm_fits/sigma);
    w_rand = ones(NP,1);
    w_cons = exp(-(norm_cons + norm_fits)/sigma);
    
    % Normalize weights
    w_total = w_elite + w_feas + w_rand + w_cons + eps;
    w_elite = w_elite ./ w_total;
    w_feas = w_feas ./ w_total;
    w_rand = w_rand ./ w_total;
    w_cons = w_cons ./ w_total;
    
    % Adaptive scaling factor with stronger adaptation
    F = 0.3 + 0.5*rand(NP,1) .* (1 - norm_fits).^0.7;
    
    % Vectorized mutation
    offspring = popdecs + repmat(F,1,D).* ...
               (repmat(w_elite,1,D).*d_elite + repmat(w_feas,1,D).*d_feas + ...
                repmat(w_rand,1,D).*d_rand + repmat(w_cons,1,D).*d_cons);
    
    % Improved boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    eta = 0.1 + 0.2*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + eta.*(ub - offspring)).*out_low + ...
               (ub - eta.*(offspring - lb)).*out_high;
    
    % Final boundary enforcement with adaptive perturbation
    perturb = 0.01*(ub-lb).*randn(NP,D).*repmat(1-norm_fits,1,D);
    offspring = max(min(offspring + perturb, ub), lb);
end
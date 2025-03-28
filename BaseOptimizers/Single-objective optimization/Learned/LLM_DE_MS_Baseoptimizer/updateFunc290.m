% MATLAB Code
function [offspring] = updateFunc290(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    abs_cons = max(0, cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons) + eps;
    norm_cons = (abs_cons - c_min) / (c_max - c_min + eps);
    
    % Process fitness (minimization)
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
    
    % Generate random pairs
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(same_idx)
        r1(same_idx) = randi(NP, sum(same_idx), 1);
        r2(same_idx) = randi(NP, sum(same_idx), 1);
        same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Adaptive parameters
    sigma_c = 0.25 + 0.1*rand();
    sigma_f = 0.25 + 0.1*rand();
    
    % Calculate weights
    exp_cons = exp(-norm_cons/sigma_c);
    exp_fits = exp(-norm_fits/sigma_f);
    w_total = exp_cons + exp_fits + 0.1*rand(NP,1);
    
    w1 = (0.6*exp_cons + 0.2*exp_fits) ./ w_total;
    w2 = (0.3*exp_cons + 0.4*exp_fits) ./ w_total;
    w3 = 0.1*ones(NP,1) ./ w_total;
    w4 = (0.4*exp_cons + 0.1*exp_fits) ./ w_total;
    
    % Scaling factor
    F = 0.5 + 0.2*rand(NP,1);
    
    % Vectorized mutation
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    d_cons = repmat(lowcons, NP, 1) - popdecs;
    
    offspring = popdecs + repmat(F,1,D).* ...
               (repmat(w1,1,D).*d_elite + repmat(w2,1,D).*d_feas + ...
               repmat(w3,1,D).*d_rand + repmat(w4,1,D).*d_cons);
    
    % Boundary handling
    out_low = offspring < lb;
    out_high = offspring > ub;
    eta = 0.1 + 0.2*rand(NP,D);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + eta.*(ub - offspring)).*out_low + ...
               (ub - eta.*(offspring - lb)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end
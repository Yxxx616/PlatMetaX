% MATLAB Code
function [offspring] = updateFunc1109(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute adaptive weights
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_std = std(abs(cons)) + eps;
    
    w_f = 1 ./ (1 + exp((popfits - f_mean)./f_std));
    w_c = 1 ./ (1 + exp(-abs(cons)./c_std));
    alpha = w_f .* w_c;
    
    % 2. Identify best/worst and elite/violated
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    k_elite = max(2, floor(NP*0.2));
    k_viol = max(2, floor(NP*0.2));
    
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k_elite);
    
    [~, sort_cons] = sort(abs(cons), 'descend');
    viol_idx = sort_cons(1:k_viol);
    
    % 3. Compute elite and repair directions
    alpha_elite = alpha(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* alpha_elite, 1) ./ (sum(alpha_elite) + eps);
    
    alpha_viol = alpha(viol_idx);
    d_repair = sum((popdecs(viol_idx,:) - x_worst) .* (1-alpha_viol), 1) ./ (sum(1-alpha_viol) + eps);
    
    % 4. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Adaptive mutation with new scaling factors
    F_e = 0.6 * alpha;
    F_r = 0.3 * (1 - alpha);
    F_d = 0.1;
    
    mutants = popdecs + F_e.*d_elite + F_r.*d_repair + F_d.*d_div;
    
    % 6. Exponential crossover with improved adaptive CR
    CR = 0.85 + 0.1 * alpha;
    offspring = popdecs;
    
    for i = 1:NP
        start = randi(D);
        L = 0;
        while rand() < CR(i) && L < D
            pos = mod(start + L - 1, D) + 1;
            offspring(i,pos) = mutants(i,pos);
            L = L + 1;
        end
    end
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
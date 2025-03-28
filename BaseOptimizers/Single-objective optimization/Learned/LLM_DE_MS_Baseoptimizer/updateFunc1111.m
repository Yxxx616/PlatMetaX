% MATLAB Code
function [offspring] = updateFunc1111(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute adaptive weights
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_std = std(abs(cons)) + eps;
    
    w = 1 ./ (1 + exp((popfits - f_mean)./f_std)) .* ...
        1 ./ (1 + exp(abs(cons)./c_std));
    
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
    
    % 3. Compute directional vectors
    w_elite = w(elite_idx);
    d_best = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_viol = w(viol_idx);
    d_repair = sum((popdecs(viol_idx,:) - x_worst) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 4. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Adaptive mutation
    F1 = 0.6;
    F2 = 0.3;
    F3 = 0.1;
    
    w_rep = repmat(w, 1, D);
    mutants = popdecs + F1.*w_rep.*repmat(d_best, NP, 1) + ...
              F2.*(1-w_rep).*repmat(d_repair, NP, 1) + ...
              F3.*d_div;
    
    % 6. Exponential crossover with adaptive CR
    CR = 0.85 + 0.1*w;
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
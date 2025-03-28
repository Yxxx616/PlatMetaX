% MATLAB Code
function [offspring] = updateFunc166(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Combined ranking with adaptive lambda
    max_cons = max(abs(cons));
    lambda = 0.5 + 0.5*(max_cons > 0);
    scores = popfits + lambda*abs(cons);
    [~, sorted_idx] = sort(scores);
    
    % 2. Population partitioning
    elite_num = round(0.3*NP);
    poor_num = round(0.3*NP);
    E = sorted_idx(1:elite_num);
    M = sorted_idx(elite_num+1:end-poor_num);
    P = sorted_idx(end-poor_num+1:end);
    
    % 3. Identify best individual (constraint-aware)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(abs(cons));
        x_best = popdecs(best_idx,:);
    end
    
    % 4. Generate random indices (vectorized)
    rand_idx = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 3));
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); r3 = rand_idx(:,3);
    
    % 5. Adaptive mutation strategies
    v = zeros(NP, D);
    
    % Elites - directed search
    F_e = 0.5 + 0.4*rand(length(E),1);
    v(E,:) = popdecs(E,:) + F_e.*(x_best - popdecs(E,:)) + ...
             0.5*(popdecs(r1(E),:) - popdecs(r2(E),:));
    
    % Moderate - balanced exploration
    w_M = abs(cons(M))/max_cons;
    F_m = 0.7*(1-w_M);
    v(M,:) = popdecs(r1(M),:) + F_m.*(popdecs(r2(M),:) - popdecs(r3(M),:)) + ...
             0.3*randn(length(M), D);
    
    % Poor - diversity enhancement
    F_p = 1.2*rand(length(P),1);
    v(P,:) = x_best + F_p.*randn(length(P), D).*(ub-lb);
    
    % 6. Constraint-aware crossover
    norm_cons = abs(cons)/max(abs(cons));
    CR = 0.9*(1 - norm_cons) + 0.1;
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    reflect_factor = 0.5 + 0.5*rand(NP, D);
    offspring(below_lb) = lb_rep(below_lb) + reflect_factor(below_lb).*...
                         (lb_rep(below_lb) - offspring(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - reflect_factor(above_ub).*...
                         (offspring(above_ub) - ub_rep(above_ub));
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
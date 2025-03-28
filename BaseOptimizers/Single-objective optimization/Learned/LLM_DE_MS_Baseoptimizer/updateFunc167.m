% MATLAB Code
function [offspring] = updateFunc167(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware ranking
    lambda = 2; % Fixed for this iteration
    penalty = max(0, cons);
    scores = popfits + lambda * penalty;
    
    % 2. Population partitioning
    [~, sorted_idx] = sort(scores);
    elite_num = round(0.3*NP);
    middle_num = round(0.4*NP);
    E = sorted_idx(1:elite_num);
    M = sorted_idx(elite_num+1:elite_num+middle_num);
    P = sorted_idx(elite_num+middle_num+1:end);
    
    % 3. Identify best individual (feasible preferred)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(penalty);
        x_best = popdecs(best_idx,:);
    end
    
    % 4. Generate random indices (vectorized)
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 4));
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % 5. Adaptive mutation strategies
    v = zeros(NP, D);
    
    % Elites - directed exploration
    v(E,:) = repmat(x_best, length(E), 1) + ...
             0.8*(popdecs(r1(E),:) - popdecs(r2(E),:)) + ...
             0.4*(popdecs(r3(E),:) - popdecs(r4(E),:));
    
    % Middle - balanced search
    F_m = 0.5 + 0.3*rand(length(M),1);
    v(M,:) = popdecs(M,:) + F_m.*(repmat(x_best, length(M),1) - popdecs(M,:)) + ...
             0.5*(popdecs(r1(M),:) - popdecs(r2(M),:));
    
    % Poor - diversity enhancement
    sigma = 0.2*(ub - lb);
    v(P,:) = popdecs(r1(P),:) + ...
             (0.5 + 0.5*rand(length(P),1)).*(popdecs(r2(P),:) - popdecs(r3(P),:)) + ...
             sigma.*randn(length(P), D);
    
    % 6. Dynamic crossover
    norm_scores = (scores - min(scores)) / (max(scores) - min(scores) + eps);
    CR = 0.9 - 0.5*norm_scores;
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
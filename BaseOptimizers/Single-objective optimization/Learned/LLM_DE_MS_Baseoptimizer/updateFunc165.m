% MATLAB Code
function [offspring] = updateFunc165(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Combined ranking and weighting
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(abs(cons));
    alpha = 0.6; % Fitness vs constraint balance
    w = alpha*(fit_rank/NP) + (1-alpha)*(cons_rank/NP);
    w = w / sum(w); % Normalize weights
    
    % 2. Population classification
    [~, sorted_idx] = sort(w, 'descend');
    elite_num = round(0.3*NP);
    poor_num = round(0.3*NP);
    E = sorted_idx(1:elite_num);
    M = sorted_idx(elite_num+1:end-poor_num);
    P = sorted_idx(end-poor_num+1:end);
    
    % 3. Identify best individual considering constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx),:);
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
    
    % 5. Mutation strategies
    v = zeros(NP, D);
    
    % For elites - focused exploitation
    F1 = 0.9 * w(E);
    F2 = 0.5 * (1-w(E));
    v(E,:) = popdecs(E,:) + F1.*(x_best - popdecs(E,:)) + ...
             F2.*(popdecs(r1(E),:) - popdecs(r2(E),:));
    
    % For moderate solutions - balanced exploration
    F = 0.7 * (1-w(M));
    epsilon = 0.3 * (1-w(M));
    v(M,:) = popdecs(r1(M),:) + F.*(popdecs(r2(M),:) - popdecs(r3(M),:)) + ...
             epsilon.*randn(length(M), D);
    
    % For poor solutions - diversity maintenance
    sigma = 0.6 * (1 + rand(length(P),1));
    v(P,:) = x_best + sigma.*randn(length(P), D);
    
    % 6. Crossover with adaptive CR
    CR = 0.8 * w + 0.2;
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clipping to ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
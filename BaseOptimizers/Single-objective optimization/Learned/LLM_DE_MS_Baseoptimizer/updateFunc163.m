% MATLAB Code
function [offspring] = updateFunc163(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Combined ranking and weighting
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(abs(cons));
    alpha = 0.7; % Fitness vs constraint balance
    w = alpha*(fit_rank/NP) + (1-alpha)*(cons_rank/NP);
    w = w / sum(w); % Normalize weights
    
    % 2. Population classification
    [~, sorted_idx] = sort(w, 'descend');
    elite_num = round(0.2*NP);
    poor_num = round(0.2*NP);
    E = sorted_idx(1:elite_num);
    M = sorted_idx(elite_num+1:end-poor_num);
    P = sorted_idx(end-poor_num+1:end);
    
    % 3. Identify best individual
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
    
    % 5. Mutation strategies
    v = zeros(NP, D);
    
    % For elites
    F1 = 0.8 * w(E);
    F2 = 0.4 * (1-w(E));
    term1 = repmat(F1,1,D) .* (repmat(x_best,length(E),1) - popdecs(E,:));
    term2 = repmat(F2,1,D) .* (popdecs(r1(E),:) - popdecs(r2(E),:));
    v(E,:) = popdecs(E,:) + term1 + term2;
    
    % For moderate solutions
    F = 0.5 * (1-w(M));
    epsilon = 0.2 * (1-w(M));
    diff = popdecs(r2(M),:) - popdecs(r3(M),:);
    v(M,:) = popdecs(r1(M),:) + repmat(F,1,D).*diff + ...
             repmat(epsilon,1,D).*randn(length(M),D);
    
    % For poor solutions
    sigma = 0.5 * (1 + rand(length(P),1));
    v(P,:) = repmat(x_best,length(P),1) + ...
             repmat(sigma,1,D).*randn(length(P),D);
    
    % 6. Crossover with adaptive CR
    CR = 0.9 * w + 0.1;
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
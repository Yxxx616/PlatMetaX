% MATLAB Code
function [offspring] = updateFunc168(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware weighting
    penalty = max(0, cons);
    w = 1 ./ (1 + penalty.^2);
    w = w / max(w); % Normalize to [0,1]
    
    % 2. Identify best individual (feasible preferred)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(penalty);
        x_best = popdecs(best_idx,:);
    end
    
    % 3. Generate random indices (vectorized)
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 4));
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % 4. Adaptive mutation strategies
    v = zeros(NP, D);
    sigma = 0.2 * (ub - lb) .* (1 - w');
    
    % Elite group (w > 0.8)
    elite = w > 0.8;
    if any(elite)
        F1 = 0.8 * w(elite);
        F2 = 0.4 * (1 - w(elite));
        v(elite,:) = repmat(x_best, sum(elite), 1) + ...
                    F1 .* (popdecs(r1(elite),:) - popdecs(r2(elite),:)) + ...
                    F2 .* (popdecs(r3(elite),:) - popdecs(r4(elite),:));
    end
    
    % Middle group (0.3 <= w <= 0.8)
    middle = w >= 0.3 & w <= 0.8;
    if any(middle)
        v(middle,:) = popdecs(middle,:) + ...
                     w(middle)' .* (repmat(x_best, sum(middle), 1) - popdecs(middle,:)) + ...
                     (1-w(middle))' .* (popdecs(r1(middle),:) - popdecs(r2(middle),:));
    end
    
    % Poor group (w < 0.3)
    poor = w < 0.3;
    if any(poor)
        v(poor,:) = popdecs(r1(poor),:) + ...
                   0.5 * (popdecs(r2(poor),:) - popdecs(r3(poor),:)) + ...
                   sigma(poor,:) .* randn(sum(poor), D);
    end
    
    % 5. Dynamic crossover
    CR = 0.9 - 0.5 * (w - min(w)) / (max(w) - min(w) + eps);
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 6. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    reflect_factor = 0.3 + 0.7 * rand(NP, D);
    offspring(below_lb) = lb_rep(below_lb) + reflect_factor(below_lb) .* ...
                         (lb_rep(below_lb) - offspring(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - reflect_factor(above_ub) .* ...
                         (offspring(above_ub) - ub_rep(above_ub));
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
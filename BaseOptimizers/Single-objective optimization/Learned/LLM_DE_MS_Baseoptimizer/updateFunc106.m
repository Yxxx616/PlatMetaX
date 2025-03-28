% MATLAB Code
function [offspring] = updateFunc106(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % 1. Boundary definitions
    lb = -1000 * ones(1, D);
    ub = 1000 * ones(1, D);
    
    % 2. Normalize constraints and fitness
    c_abs = abs(cons);
    c_max = max(c_abs);
    norm_cons = c_abs / (c_max + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    % 3. Select elite individuals (top 3)
    beta = 0.5;
    elite_scores = popfits + beta * c_abs;
    [~, elite_idx] = sort(elite_scores);
    elite_idx = elite_idx(1:3);
    elites = popdecs(elite_idx, :);
    x_base = mean(elites, 1);
    
    % 4. Create opposition population
    x_opp = lb + ub - x_base;
    
    % 5. Generate mutation vectors
    v = zeros(NP, D);
    for i = 1:NP
        % Select 3 distinct random individuals (excluding current and elites)
        candidates = setdiff(1:NP, [i, elite_idx']);
        idx = randperm(length(candidates), 3);
        r1 = candidates(idx(1)); 
        r2 = candidates(idx(2));
        r3 = candidates(idx(3));
        
        % Adaptive scaling factors
        F1 = 0.5 + 0.5 * (1 - norm_fits(i));
        F2 = 0.3 + 0.7 * (1 - norm_cons(i));
        
        % Mutation with opposition-based learning
        v(i,:) = x_base + F1*(popdecs(r1,:)-popdecs(r2,:)) + F2*(popdecs(r3,:)-x_opp);
    end
    
    % 6. Dynamic crossover rate with jitter
    CR = 0.2 + 0.6 * (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % 7. Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 8. Boundary control with reflection
    below_lb = offspring < lb;
    above_ub = offspring > ub;
    offspring(below_lb) = 2*lb(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub(above_ub) - offspring(above_ub);
    
    % 9. Final clamping
    offspring = min(max(offspring, lb), ub);
end
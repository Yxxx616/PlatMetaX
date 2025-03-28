% MATLAB Code
function [offspring] = updateFunc104(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % 1. Calculate constraint violation magnitude
    c_abs = abs(cons);
    c_max = max(c_abs);
    norm_cons = c_abs / (c_max + eps);
    
    % 2. Normalize fitness
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    % 3. Select elite individuals (top 3)
    beta = 0.7;
    elite_scores = popfits + beta * c_abs;
    [~, elite_idx] = sort(elite_scores);
    elite_idx = elite_idx(1:3);
    elites = popdecs(elite_idx, :);
    
    % 4. Calculate elite weights using sigmoid function
    gamma = 0.5;
    elite_weights = 1./(1 + exp(-(popfits(elite_idx) + gamma*cons(elite_idx))));
    elite_weights = elite_weights / sum(elite_weights);
    
    % 5. Base vector as weighted combination of elites
    x_base = elite_weights' * elites;
    
    % 6. Generate mutation vectors
    v = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct random individuals
        candidates = setdiff(1:NP, i);
        idx = randperm(length(candidates), 4);
        r1 = candidates(idx(1)); r2 = candidates(idx(2));
        r3 = candidates(idx(3)); r4 = candidates(idx(4));
        
        % Adaptive scaling factors
        F1 = 0.5 + 0.5 * tanh(1 - norm_cons(i));
        F2 = 0.3 + 0.7 * tanh(1 - norm_fits(i));
        
        % Mutation
        v(i,:) = x_base + F1*(popdecs(r1,:)-popdecs(r2,:)) + F2*(popdecs(r3,:)-popdecs(r4,:));
    end
    
    % 7. Dynamic crossover rate
    CR = 0.1 + 0.8 * (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    % 8. Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % 9. Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 10. Boundary control with reflection
    lb = -1000 * ones(1,D);
    ub = 1000 * ones(1,D);
    
    % Reflection for lower bound
    below_lb = offspring < lb;
    offspring(below_lb) = 2*lb(below_lb) - offspring(below_lb);
    
    % Reflection for upper bound
    above_ub = offspring > ub;
    offspring(above_ub) = 2*ub(above_ub) - offspring(above_ub);
    
    % Final clamping to ensure within bounds
    offspring = min(max(offspring, lb), ub);
end
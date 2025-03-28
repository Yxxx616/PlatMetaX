% MATLAB Code
function [offspring] = updateFunc1122(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / (c_max + eps);
    
    % 2. Compute adaptive weights
    w = 1 ./ (1 + exp(-5*(0.7*norm_fit + 0.3*norm_cons)));
    w = w(:);
    
    % 3. Identify elite and anti-elite groups
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    k = max(2, floor(NP*0.2));
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k);
    anti_idx = sort_fit(end-k+1:end);
    
    % 4. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_anti = w(anti_idx);
    d_anti = sum((x_worst - popdecs(anti_idx,:)) .* (1-w_anti), 1) / (sum(1-w_anti) + eps);
    
    % 5. Generate diversity component
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 6. Hybrid mutation
    F1 = 0.7 * w;
    F2 = 0.3 * (1 - w);
    F3 = 0.5 * (1 - w);
    mutants = popdecs + F1.*d_elite + F2.*d_anti + F3.*d_div;
    
    % 7. Adaptive crossover
    CR = 0.9 - 0.4 * tanh(3*(w + norm_cons));
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling
    out_of_bounds = offspring < lb | offspring > ub;
    rand_vals = lb + (ub-lb).*rand(NP,D);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
    offspring = min(max(offspring, lb), ub);
end
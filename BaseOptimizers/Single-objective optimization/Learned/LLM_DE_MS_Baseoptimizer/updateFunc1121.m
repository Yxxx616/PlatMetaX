% MATLAB Code
function [offspring] = updateFunc1121(popdecs, popfits, cons)
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
    w = 1 ./ (1 + exp(-5*(0.6*norm_fit + 0.4*norm_cons)));
    w = w(:);
    
    % 3. Identify elite and anti-elite groups
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    k_elite = max(2, floor(NP*0.15));
    k_anti = max(2, floor(NP*0.15));
    
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k_elite);
    anti_idx = sort_fit(end-k_anti+1:end);
    
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
    F1 = 0.6 + 0.2*w;
    F2 = 0.4*(1-w);
    mutants = popdecs + F1.*d_elite + (1-F1).*d_anti + F2.*d_div;
    
    % 7. Adaptive crossover
    CR = 0.85 - 0.35*tanh(5*(w + norm_cons));
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
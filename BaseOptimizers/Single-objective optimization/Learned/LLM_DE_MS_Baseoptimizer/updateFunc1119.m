% MATLAB Code
function [offspring] = updateFunc1119(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    
    % 2. Sigmoid weighting combining fitness and constraints
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / (c_max + eps);
    w = 1 ./ (1 + exp(-5*(0.6*norm_fit + 0.4*norm_cons)));
    w = w(:);
    
    % 3. Identify elite and violated individuals
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    k_elite = max(2, floor(NP*0.3));
    k_viol = max(2, floor(NP*0.3));
    
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k_elite);
    
    [~, sort_cons] = sort(abs(cons), 'descend');
    viol_idx = sort_cons(1:k_viol);
    
    % 4. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_viol = w(viol_idx);
    d_viol = sum((x_worst - popdecs(viol_idx,:)) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 5. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 6. Hybrid mutation with adaptive weights
    F1 = 0.8 * (1 - norm_fit);
    F2 = 0.6 * norm_cons;
    F3 = 0.4 * ones(NP, 1);
    
    w_rep = repmat(w, 1, D);
    mutants = popdecs + F1.*w_rep.*repmat(d_elite, NP, 1) + ...
              F2.*(1-w_rep).*repmat(d_viol, NP, 1) + ...
              F3.*d_div;
    
    % 7. Dynamic crossover based on fitness
    CR = 0.9 - 0.5*norm_fit;
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with random reinitialization
    out_of_bounds = offspring < lb | offspring > ub;
    rand_vals = lb + (ub-lb).*rand(NP,D);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1120(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / (c_max + eps);
    
    % 2. Combined weights using sigmoid
    w = 1 ./ (1 + exp(-5*(0.7*norm_fit + 0.3*norm_cons)));
    w = w(:);
    
    % 3. Identify elite and violated individuals
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    k_elite = max(2, floor(NP*0.2));
    k_viol = max(2, floor(NP*0.2));
    
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k_elite);
    
    [~, sort_cons] = sort(abs(cons), 'descend');
    viol_idx = sort_cons(1:k_viol);
    
    % 4. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_viol = w(viol_idx);
    d_viol = sum((x_worst - popdecs(viol_idx,:)) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 5. Generate diversity vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:) + 0.1*(x_best - x_worst);
    
    % 6. Adaptive scaling factors
    F1 = 0.5 + 0.5./(1 + exp(-5*(0.7*norm_fit + 0.3*norm_cons)));
    
    % 7. Hybrid mutation
    w_rep = repmat(w, 1, D);
    F1_rep = repmat(F1, 1, D);
    mutants = popdecs + F1_rep.*repmat(d_elite, NP, 1) + ...
              (1-F1_rep).*repmat(d_viol, NP, 1) + ...
              0.5*d_div;
    
    % 8. Adaptive crossover
    CR = 0.9 - 0.4*tanh(5*(norm_fit + norm_cons));
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling
    out_of_bounds = offspring < lb | offspring > ub;
    rand_vals = lb + (ub-lb).*rand(NP,D);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1117(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Adaptive weight calculation
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    w = 0.6*(1 - (popfits-f_min)/(f_max-f_min+eps)) + ...
        0.4*(1 - abs(cons)/(c_max+eps));
    w = w(:);
    
    % 2. Identify elite and violated individuals
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    k_elite = max(2, floor(NP*0.15));
    k_viol = max(2, floor(NP*0.15));
    
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k_elite);
    
    [~, sort_cons] = sort(abs(cons), 'descend');
    viol_idx = sort_cons(1:k_viol);
    
    % 3. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_viol = w(viol_idx);
    d_viol = sum((x_worst - popdecs(viol_idx,:)) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 4. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Hybrid mutation with adaptive weights
    F1 = 0.8; F2 = 0.6; F3 = 0.4;
    w_rep = repmat(w, 1, D);
    mutants = popdecs + F1.*w_rep.*repmat(d_elite, NP, 1) + ...
              F2.*(1-w_rep).*repmat(d_viol, NP, 1) + ...
              F3.*d_div;
    
    % 6. Dynamic crossover
    CR = 0.85 + 0.15*(popfits-f_min)/(f_max-f_min+eps);
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Reflective boundary handling
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
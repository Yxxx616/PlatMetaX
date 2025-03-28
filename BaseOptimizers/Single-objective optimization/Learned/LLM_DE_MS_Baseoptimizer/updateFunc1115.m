% MATLAB Code
function [offspring] = updateFunc1115(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute adaptive weights combining fitness and constraints
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    w = 1 - 0.6*f_norm - 0.4*c_norm;  % Adjusted weights
    w = w(:);
    
    % 2. Identify best/worst and elite/violated individuals
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
    
    % 3. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_viol = w(viol_idx);
    d_opp = sum((x_worst - popdecs(viol_idx,:)) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 4. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Enhanced adaptive mutation with dynamic scaling
    F1 = 0.8;  % Increased exploitation
    F2 = 0.4;  % Balanced constraint handling
    F3 = 0.3;  % Maintained diversity
    
    w_rep = repmat(w, 1, D);
    mutants = popdecs + F1.*w_rep.*repmat(d_elite, NP, 1) + ...
              F2.*(1-w_rep).*repmat(d_opp, NP, 1) + ...
              F3.*d_div;
    
    % 6. Dynamic binomial crossover
    CR = 0.85 + 0.15*rand(NP,1);  % More dynamic CR
    mask = rand(NP,D) < CR;
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Improved boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with random component
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    r = rand(NP,D);
    offspring(below) = lb_rep(below) + r(below).*(popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - r(above).*(ub_rep(above) - popdecs(above));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
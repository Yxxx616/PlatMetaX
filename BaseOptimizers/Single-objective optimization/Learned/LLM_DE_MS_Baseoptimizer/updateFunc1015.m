% MATLAB Code
function [offspring] = updateFunc1015(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    sigma_f = std(popfits) + 1e-12;
    sigma_c = std(cons) + 1e-12;
    f_norm = popfits / sigma_f;
    c_norm = cons / sigma_c;
    
    % Adaptive weights combining fitness and constraints
    w = 1./(1 + exp(-(f_norm + c_norm)));
    w = w / max(w);
    
    % Elite vector (weighted centroid)
    x_elite = sum(popdecs .* w, 1) / sum(w);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive scaling factor based on constraints
    F = 0.5 + 0.3 * tanh(5 * c_norm);
    
    % Elite-guided mutation
    elite_term = x_elite - popdecs;
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    v = popdecs + F .* (elite_term + diff_term);
    
    % Directional perturbation
    f_diff = popfits(r1) - popfits(r2);
    dir_sign = sign(f_diff);
    eta = 0.1 * randn(NP, 1);
    delta = eta .* dir_sign .* diff_term;
    
    % Final mutation with adaptive blending
    mutants = v + delta .* (1 - w);
    
    % Dynamic crossover
    CR = 0.9 * w + 0.1;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
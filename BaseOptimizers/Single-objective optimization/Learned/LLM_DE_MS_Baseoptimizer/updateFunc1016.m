% MATLAB Code
function [offspring] = updateFunc1016(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalization factors
    sigma_f = std(popfits) + 1e-12;
    sigma_c = std(cons) + 1e-12;
    
    % Weight calculation combining fitness and constraints
    w = 1./(1 + exp(-(popfits/sigma_f + cons/sigma_c)));
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
    F = 0.5 + 0.2 * tanh(5 * cons/sigma_c);
    
    % Constraint-guided mutation
    elite_term = x_elite - popdecs;
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    f_diff = popfits(r1) - popfits(r2);
    eta = 0.1 * randn(NP, 1);
    dir_term = eta .* sign(f_diff) .* diff_term;
    
    mutants = popdecs + F .* (elite_term + diff_term) + dir_term;
    
    % Dynamic crossover
    CR = 0.85 + 0.1 * w;
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
% MATLAB Code
function [offspring] = updateFunc1018(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalization factors
    sigma_f = std(popfits) + 1e-12;
    sigma_c = std(cons) + 1e-12;
    mean_f = mean(popfits);
    mean_c = mean(cons);
    
    % Weight calculation (balancing fitness and constraints)
    alpha = 0.8; beta = 0.4;
    w = 1./(1 + exp(-(alpha*(popfits-mean_f)/sigma_f - beta*(cons-mean_c)/sigma_c));
    w = w / sum(w);
    
    % Elite vector (weighted centroid)
    x_elite = sum(popdecs .* w, 1);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive scaling factor
    F = 0.5 + 0.2 * tanh(5 * (cons-mean_c)/sigma_c);
    
    % Direction vectors
    dir_sign = sign(popfits - mean_f);
    dir_vectors = (x_elite - popdecs) .* dir_sign;
    
    % Mutation with adaptive components
    eta = 0.1 * randn(NP, 1);
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F .* dir_vectors + eta .* diff_vectors;
    
    % Dynamic crossover based on constraint violation
    max_cons = max(abs(cons)) + 1e-12;
    CR = 0.9 - 0.4 * (abs(cons)/max_cons);
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
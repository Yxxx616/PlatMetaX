% MATLAB Code
function [offspring] = updateFunc1012(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_mean = mean(popfits); f_std = std(popfits);
    c_mean = mean(cons); c_std = std(cons);
    
    % Calculate adaptive weights
    w_f = 1./(1 + exp(-5*(popfits - f_mean)/max(f_std,1e-6)));
    w_c = 1./(1 + exp(-5*(cons - c_mean)/max(c_std,1e-6)));
    w = w_f .* w_c;
    w = w / max(w);
    
    % Elite vector (weighted centroid)
    x_elite = sum(bsxfun(@times, popdecs, w), 1) / sum(w);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive scaling factor
    F = 0.5 * (1 + tanh(5*(cons - c_mean)/max(c_std,1e-6)));
    
    % Direction vectors
    use_elite = rand(NP,1) < w;
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    elite_term = bsxfun(@minus, x_elite, popdecs);
    d = bsxfun(@times, use_elite, elite_term) + bsxfun(@times, ~use_elite, diff_term);
    
    % Mutation with adaptive noise
    rand_term = 0.1 * (1 - w) .* randn(NP, D);
    mutants = popdecs + bsxfun(@times, F, d) + rand_term;
    
    % Dynamic crossover rate
    CR = 0.7 * w + 0.3 * (1 - cons/(max(cons) + eps));
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    lb_viol = offspring < lb_matrix;
    offspring(lb_viol) = 2*lb_matrix(lb_viol) - offspring(lb_viol);
    
    ub_viol = offspring > ub_matrix;
    offspring(ub_viol) = 2*ub_matrix(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
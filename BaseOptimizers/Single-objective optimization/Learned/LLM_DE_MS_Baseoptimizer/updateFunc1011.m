% MATLAB Code
function [offspring] = updateFunc1011(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate adaptive weights
    med_f = median(popfits);
    med_c = median(cons);
    w = 1./(1 + exp(5*(popfits - med_f))) .* 1./(1 + exp(5*(cons - med_c)));
    w = w / max(w); % Normalize to [0,1]
    
    % Elite vector (weighted centroid)
    x_elite = sum(bsxfun(@times, popdecs, w), 1) / sum(w);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive scaling factor
    c_min = min(cons);
    c_max = max(cons);
    F = 0.4 + 0.3 * (cons - c_min) / (c_max - c_min + eps);
    
    % Direction vectors
    use_elite = rand(NP,1) < w;
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    elite_term = bsxfun(@minus, x_elite, popdecs);
    d = bsxfun(@times, use_elite, elite_term) + bsxfun(@times, ~use_elite, diff_term);
    
    % Mutation
    rand_term = 0.2 * (1 - w) .* randn(NP, D);
    mutants = popdecs + bsxfun(@times, F, d) + rand_term;
    
    % Dynamic crossover rate
    CR = 0.5 + 0.3 * w + 0.2 * (1 - cons/(c_max + eps));
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with midpoint reflection
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    lb_viol = offspring < lb_matrix;
    offspring(lb_viol) = (popdecs(lb_viol) + lb_matrix(lb_viol)) / 2;
    
    ub_viol = offspring > ub_matrix;
    offspring(ub_viol) = (popdecs(ub_viol) + ub_matrix(ub_viol)) / 2;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
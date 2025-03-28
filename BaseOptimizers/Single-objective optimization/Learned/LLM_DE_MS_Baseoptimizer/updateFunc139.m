% MATLAB Code
function [offspring] = updateFunc139(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Sort by combined score (60% fitness, 40% constraints)
    combined_scores = 0.6 * norm_fits + 0.4 * (1 - norm_cons);
    [~, sorted_idx] = sort(combined_scores);
    
    % Select indices
    best_idx = sorted_idx(1);
    elite_size = max(3, floor(0.25*NP));
    elite_idx = sorted_idx(1:elite_size);
    worst_idx = sorted_idx(end-elite_size+1:end);
    
    % Calculate centroids
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_worst = mean(popdecs(worst_idx,:), 1);
    x_best = popdecs(best_idx,:);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Adaptive parameters
    F1 = 0.6 * (1 - norm_cons) + 0.2 * norm_fits;
    F2 = 0.3 + 0.4 * rand(NP, 1);
    F3 = 0.2 + 0.5 * norm_cons;
    CR = 0.85 * (1 - norm_cons) + 0.1 * norm_fits;
    
    % Elite direction vector
    v_elite = repmat(x_best, NP, 1) + ...
              F1 .* (repmat(x_elite, NP, 1) - repmat(x_worst, NP, 1)) + ...
              F2 .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Random direction vector
    v_rand = popdecs(r1,:) + F3 .* (popdecs(r2,:) - popdecs(r3,:));
    
    % Weighted combination
    w = norm_fits ./ (norm_fits + norm_cons + eps);
    sigma = 0.15 * (1 + norm_cons);
    v = repmat(w, 1, D) .* v_elite + ...
        repmat(1-w, 1, D) .* v_rand + ...
        sigma .* randn(NP, D);
    
    % Crossover with guaranteed dimension
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    reflect_factor = 0.4 + 0.4 * repmat(norm_fits, 1, D) .* rand(NP, D);
    
    offspring(below_lb) = lb_rep(below_lb) + reflect_factor(below_lb) .* ...
                        (ub_rep(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - reflect_factor(above_ub) .* ...
                        (ub_rep(above_ub) - lb_rep(above_ub));
    
    % Final boundary check
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end
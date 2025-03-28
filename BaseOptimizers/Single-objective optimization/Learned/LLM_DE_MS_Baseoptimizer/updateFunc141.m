% MATLAB Code
function [offspring] = updateFunc141(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalization with protection against constant values
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Combined score (higher is better)
    scores = 0.6 * (1 - norm_fits) + 0.4 * (1 - norm_cons);
    
    % Population partitioning
    [~, sorted_idx] = sort(scores);
    elite_size = floor(0.2 * NP);
    mod_size = floor(0.5 * NP);
    elite_idx = sorted_idx(1:elite_size);
    mod_idx = sorted_idx(elite_size+1:elite_size+mod_size);
    poor_idx = sorted_idx(elite_size+mod_size+1:end);
    
    % Centroid calculations
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_mod = mean(popdecs(mod_idx,:), 1);
    x_poor = mean(popdecs(poor_idx,:), 1);
    
    % Random indices matrix
    rand_idx = randi(NP, NP, 7);
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); r3 = rand_idx(:,3);
    r4 = rand_idx(:,4); r5 = rand_idx(:,5); r6 = rand_idx(:,6);
    r7 = rand_idx(:,7);
    
    % Adaptive parameters
    F1 = 0.5 * (1 + norm_fits);
    F2 = 0.3 + 0.4 * rand(NP, 1);
    F3 = 0.2 * (1 - norm_cons);
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation vectors
    v_cons = F2 .* (x_mod - x_poor) .* (1 - norm_cons);
    v_div = F3 .* (popdecs(r1,:) - popdecs(r2,:)) .* norm_fits;
    
    v = zeros(NP, D);
    for i = 1:NP
        if ismember(i, elite_idx)
            v(i,:) = x_elite + F1(i) * (x_elite - x_poor) + v_cons(i,:) + v_div(i,:);
        elseif ismember(i, mod_idx)
            v(i,:) = popdecs(i,:) + v_cons(i,:) + v_div(i,:);
        else
            v(i,:) = popdecs(r3(i),:) + v_cons(i,:);
        end
    end
    
    % Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Fitness-based reflection probability
    reflect_prob = repmat(0.4 + 0.5 * norm_fits, 1, D);
    reflect_mask = rand(NP, D) < reflect_prob;
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Reflection for selected violations
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - ...
                                       offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - ...
                                       offspring(above_ub & reflect_mask);
    
    % Random reinitialization for remaining violations
    out_of_bounds = (offspring < lb_rep) | (offspring > ub_rep);
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end
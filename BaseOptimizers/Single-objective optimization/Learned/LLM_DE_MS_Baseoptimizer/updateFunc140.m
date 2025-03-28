% MATLAB Code
function [offspring] = updateFunc140(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalization
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    scores = 0.7 * norm_fits + 0.3 * (1 - norm_cons);
    
    % Population partitioning
    [~, sorted_idx] = sort(scores);
    elite_size = floor(0.3 * NP);
    mod_size = floor(0.4 * NP);
    elite_idx = sorted_idx(1:elite_size);
    mod_idx = sorted_idx(elite_size+1:elite_size+mod_size);
    poor_idx = sorted_idx(elite_size+mod_size+1:end);
    
    % Centroid calculations
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_mod = mean(popdecs(mod_idx,:), 1);
    x_poor = mean(popdecs(poor_idx,:), 1);
    
    % Random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    r5 = randi(NP, NP, 1);
    r6 = randi(NP, NP, 1);
    r7 = randi(NP, NP, 1);
    
    % Adaptive parameters
    F_base = 0.5;
    F_elite = F_base * (1 - norm_cons) + 0.3 * norm_fits;
    F_mod = 0.3 + 0.4 * rand(NP, 1);
    F_poor = 0.2 + 0.6 * norm_cons;
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation vectors
    v = zeros(NP, D);
    for i = 1:NP
        if ismember(i, elite_idx)
            v(i,:) = x_elite + F_elite(i) * (x_elite - x_poor) + ...
                     F_mod(i) * (popdecs(r1(i),:) - popdecs(r2(i),:));
        elseif ismember(i, mod_idx)
            v(i,:) = popdecs(i,:) + F_mod(i) * (x_elite - popdecs(i,:)) + ...
                     F_poor(i) * (popdecs(r3(i),:) - popdecs(r4(i),:));
        else
            v(i,:) = popdecs(r5(i),:) + F_poor(i) * ...
                     (popdecs(r6(i),:) - popdecs(r7(i),:));
        end
    end
    
    % Crossover
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with fitness-based probability
    reflect_prob = 0.3 + 0.5 * repmat(norm_fits, 1, D);
    reflect_mask = rand(NP, D) < reflect_prob;
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - ...
                                       offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - ...
                                       offspring(above_ub & reflect_mask);
    
    % Random reinitialization for remaining violations
    out_of_bounds = (offspring < lb_rep) | (offspring > ub_rep);
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end
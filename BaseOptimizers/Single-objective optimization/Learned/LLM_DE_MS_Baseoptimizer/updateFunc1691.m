% MATLAB Code
function [offspring] = updateFunc1691(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (positive values indicate violation)
    pos_cons = max(0, cons);
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    w_cons = (pos_cons - c_min) / (c_max - c_min + eps);
    
    % Normalize fitness (0 is best)
    f_min = min(popfits);
    f_max = max(popfits);
    w_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    % Rank population by combined metric
    combined_rank = 0.7*w_fit + 0.3*w_cons;
    [~, rank_idx] = sort(combined_rank);
    elite_size = max(3, round(NP*0.3));
    elite_pool = rank_idx(1:elite_size);
    best_idx = rank_idx(1);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-generate random numbers for vectorization
    F = 0.5 + 0.3*rand(NP,1);
    rand_perturb = randn(NP, D) .* (ub - lb);
    
    for i = 1:NP
        % Select base vector from elite pool
        base_idx = elite_pool(randi(elite_size));
        x_base = popdecs(base_idx, :);
        
        % Select distinct random vectors (excluding current and base)
        idx_pool = setdiff(1:NP, [i, base_idx]);
        r = idx_pool(randperm(length(idx_pool), 2));
        r1 = r(1); r2 = r(2);
        
        % Direction vectors
        best_dir = popdecs(best_idx,:) - popdecs(i,:);
        diff_vector = popdecs(r1,:) - popdecs(r2,:);
        
        % Adaptive mutation
        mutation = x_base + F(i)*diff_vector + ...
                  (0.5 + 0.3*w_fit(i))*best_dir + ...
                  0.2*(1-w_cons(i))*rand_perturb(i,:);
        
        % Exponential crossover with adaptive length
        j_rand = randi(D);
        L = max(2, 1 + floor(D*exp(-(w_fit(i)+w_cons(i)))));
        mask = false(1,D);
        mask(mod(j_rand-1:j_rand+L-2, D)+1) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with reflection
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        offspring(i,viol_low) = 2*lb(viol_low) - offspring(i,viol_low);
        offspring(i,viol_high) = 2*ub(viol_high) - offspring(i,viol_high);
    end
    
    % Final clipping and small perturbation
    offspring = min(max(offspring, lb), ub);
    perturb_mask = rand(NP,D) < 0.1;
    offspring(perturb_mask) = offspring(perturb_mask) + ...
                            0.01*(ub(perturb_mask)-lb(perturb_mask)).*randn(sum(perturb_mask(:)),1);
    offspring = min(max(offspring, lb), ub);
end
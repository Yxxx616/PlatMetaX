% MATLAB Code
function [offspring] = updateFunc1024(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + 1e-12);
    norm_cons = abs(cons) / (max(abs(cons)) + 1e-12);
    
    % Composite score (60% fitness, 40% constraints)
    scores = 0.6 * norm_fits + 0.4 * (1 - norm_cons);
    
    % Elite selection (top 30%)
    [~, sorted_idx] = sort(scores, 'descend');
    elite_num = max(1, floor(0.3*NP));
    elite_mask = false(NP,1);
    elite_mask(sorted_idx(1:elite_num)) = true;
    
    % Weighted elite centroid
    elite_weights = scores(elite_mask) / sum(scores(elite_mask));
    x_elite = sum(popdecs(elite_mask,:) .* elite_weights(:), 1);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive parameters
    F = 0.3 + 0.7 * norm_fits; % Scaling factor [0.3,1]
    eta = 0.1 * (1 - norm_cons) .* randn(NP, 1); % Constraint-aware noise
    
    % Direction vectors with fitness sign
    dir_sign = sign(popfits - median(popfits));
    dir_vectors = (x_elite - popdecs) .* dir_sign(:);
    
    % Differential vectors
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    
    % Mutation
    mutants = popdecs + F(:).*dir_vectors + eta(:).*diff_vectors;
    
    % Dynamic crossover rate
    CR = 0.9 - 0.4 * norm_fits;
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
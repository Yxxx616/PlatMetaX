% MATLAB Code
function [offspring] = updateFunc1022(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Composite score (60% fitness, 40% constraints)
    norm_fits = (popfits - mean(popfits)) / (std(popfits) + 1e-12);
    norm_cons = (cons - mean(cons)) / (std(cons) + 1e-12);
    scores = 0.6 * norm_fits + 0.4 * norm_cons;
    
    % Elite selection (top 30%)
    [~, sorted_idx] = sort(scores);
    elite_num = max(1, floor(0.3*NP));
    elite_mask = false(NP,1);
    elite_mask(sorted_idx(1:elite_num)) = true;
    
    % Inverse rank weights for elites
    elite_ranks = tiedrank(-scores(elite_mask));
    elite_weights = 1./elite_ranks;
    elite_weights = elite_weights / sum(elite_weights);
    
    % Elite centroid
    x_elite = sum(popdecs(elite_mask,:) .* elite_weights(:), 1);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive parameters
    F = 0.5 * (1 + tanh(2 * scores)); % Scaling factor [0.5,1]
    eta = 0.1 * (1 - abs(scores)) .* randn(NP, 1); % Adaptive noise
    
    % Direction vectors with fitness sign
    dir_sign = sign(popfits - mean(popfits));
    dir_vectors = (x_elite - popdecs) .* dir_sign(:);
    
    % Mutation
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F(:).*dir_vectors + eta(:).*diff_vectors;
    
    % Dynamic crossover rate
    min_s = min(scores);
    max_s = max(scores);
    CR = 0.9 - 0.5 * (scores - min_s)/(max_s - min_s + 1e-12);
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
% MATLAB Code
function [offspring] = updateFunc1021(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    mu_f = mean(popfits);
    sigma_f = std(popfits) + 1e-12;
    norm_fits = (popfits - mu_f) / sigma_f;
    
    mu_c = mean(cons);
    sigma_c = std(cons) + 1e-12;
    norm_cons = (cons - mu_c) / sigma_c;
    
    % Composite score (alpha=0.7)
    scores = 0.7 * norm_fits + 0.3 * norm_cons;
    
    % Elite selection (top 30%)
    [~, sorted_idx] = sort(scores);
    elite_num = max(1, floor(0.3*NP));
    elite_mask = false(NP,1);
    elite_mask(sorted_idx(1:elite_num)) = true;
    
    % Elite weights (rank-based)
    elite_ranks = tiedrank(-scores(elite_mask)); % higher rank for better scores
    elite_weights = elite_ranks / sum(elite_ranks);
    
    % Elite centroid
    x_elite = sum(popdecs(elite_mask,:) .* elite_weights(:), 1);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive parameters
    F = 0.4 + 0.3 * tanh(3 * abs(scores)); % Scaling factor
    eta = 0.2 * (1 - abs(scores)) .* randn(NP, 1); % Noise term
    
    % Direction vectors
    dir_sign = sign(popfits - mu_f);
    dir_vectors = (x_elite - popdecs) .* dir_sign(:);
    
    % Mutation
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F(:).*dir_vectors + eta(:).*diff_vectors;
    
    % Dynamic crossover rate
    min_s = min(scores);
    max_s = max(scores);
    CR = 0.85 - 0.4 * (scores - min_s)/(max_s - min_s + 1e-12);
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
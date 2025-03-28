% MATLAB Code
function [offspring] = updateFunc1020(popdecs, popfits, cons)
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
    
    % Composite score (lower is better)
    alpha = 0.6;
    scores = alpha * norm_fits + (1-alpha) * norm_cons;
    
    % Elite selection (top 30%)
    [~, sorted_idx] = sort(scores);
    elite_num = max(1, floor(0.3*NP));
    elite_mask = false(NP,1);
    elite_mask(sorted_idx(1:elite_num)) = true;
    
    % Calculate elite weights
    elite_ranks = tiedrank(scores(elite_mask));
    elite_weights = 1./(1 + elite_ranks);
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
    F = 0.5 + 0.3 * tanh(2 * abs(norm_cons)); % Scaling factor
    eta = 0.1 * (1 - abs(norm_cons)) .* randn(NP, 1); % Noise term
    
    % Direction vectors
    dir_sign = sign(popfits - mu_f);
    dir_vectors = (x_elite - popdecs) .* dir_sign(:);
    
    % Mutation
    diff_vectors = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F(:).*dir_vectors + eta(:).*diff_vectors;
    
    % Crossover with adaptive CR
    CR = 0.9 - 0.5 * (scores - min(scores))/(max(scores) - min(scores) + 1e-12);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with bounce-back
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = lb(lb_viol) + 0.5*rand(sum(lb_viol(:)),1).*(popdecs(lb_viol)-lb(lb_viol));
    offspring(ub_viol) = ub(ub_viol) - 0.5*rand(sum(ub_viol(:)),1).*(ub(ub_viol)-popdecs(ub_viol));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
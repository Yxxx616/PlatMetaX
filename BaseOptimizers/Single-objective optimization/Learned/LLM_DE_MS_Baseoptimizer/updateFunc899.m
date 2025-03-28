% MATLAB Code
function [offspring] = updateFunc899(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Enhanced fitness-weighted centroid
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    weights = 1 ./ (1 + exp(5*(popfits - mean_fit)/std_fit));
    weights = weights / sum(weights);
    centroid = sum(bsxfun(@times, popdecs, weights), 1);
    
    % 2. Improved constraint handling
    max_cons = max(abs(cons)) + eps;
    alpha = 1 - tanh(abs(cons)/max_cons);
    beta = 0.5 * (1 + alpha);
    
    % 3. Vectorized random indices selection
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r = candidates(randperm(length(candidates), 4));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3); r4(i) = r(4);
    end
    
    % 4. Directional perturbation with constraints
    diff = popdecs(r1,:) - popdecs(r2,:);
    norm_diff = sqrt(sum(diff.^2, 2)) + eps;
    d = bsxfun(@times, sign(cons), diff ./ norm_diff(:,ones(1,D)));
    
    % 5. Composite mutation with adaptive F
    F = 0.5 + 0.3 * rand(NP,1);
    term1 = bsxfun(@times, alpha, centroid - popdecs);
    term2 = bsxfun(@times, beta, d);
    term3 = bsxfun(@times, (1-alpha), popdecs(r3,:) - popdecs(r4,:));
    mutants = popdecs + bsxfun(@times, F, term1 + term2) + term3;
    
    % 6. Enhanced adaptive crossover
    min_fit = min(popfits);
    max_fit = max(popfits);
    CR = 0.2 + 0.6 * (popfits - min_fit) / (max_fit - min_fit + eps);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Improved boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    norm_fit = (popfits - min_fit) / (max_fit - min_fit + eps);
    reflect_factor = 0.1 * (1 - norm_fit);
    reflect_factor = repmat(reflect_factor, 1, D);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (ub - lb);
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (ub - lb);
    
    % 9. Final clamping with adaptive perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    perturbation = 0.01 * (ub - lb) .* randn(NP,D) .* (1 - norm_fit(:,ones(1,D)));
    offspring = offspring + perturbation;
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % 10. Elite preservation with local search
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, round(0.1*NP));
    elite = popdecs(sorted_idx(1:elite_size),:);
    offspring(sorted_idx(1:elite_size),:) = elite + 0.001 * (ub - lb) .* randn(elite_size, D);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
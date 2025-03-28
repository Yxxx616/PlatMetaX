% MATLAB Code
function [offspring] = updateFunc897(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weighted direction vector
    median_fit = median(popfits);
    sigma_f = std(popfits) + eps;
    weights = exp(-abs(popfits - median_fit)/sigma_f);
    weights = weights / sum(weights);
    median_pop = median(popdecs, 1);
    dw = sum(bsxfun(@times, popdecs - median_pop, weights), 1);
    
    % 2. Constraint-aware scaling factors
    max_cons = max(abs(cons)) + eps;
    alpha = 1 - tanh(abs(cons)/max_cons);
    beta = 0.1 * (1 - alpha);
    
    % 3. Initialize offspring and parameters
    offspring = popdecs;
    F = 0.7;
    
    % Generate all random indices first (vectorized)
    all_indices = 1:NP;
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        [~, sorted_idx] = sort(popfits(candidates));
        tournament_size = max(3, round(NP/5));
        selected = candidates(sorted_idx(1:tournament_size));
        r = selected(randperm(tournament_size, 3));
        r1(i) = r(1); r2(i) = r(2); r3(i) = r(3);
    end
    
    % 4. Composite mutation (vectorized)
    term1 = bsxfun(@times, alpha, dw);
    term2 = (1-alpha) .* (popdecs(r2,:) - popdecs(r3,:));
    perturbation = bsxfun(@times, tanh(abs(cons)), sign(cons)) .* randn(NP,D);
    mutants = popdecs(r1,:) + F * (term1 + term2) + bsxfun(@times, beta, perturbation);
    
    % 5. Adaptive crossover (vectorized)
    CR = 0.5 + 0.4 * (1 - sqrt((1:NP)'/NP)) .* rand(NP,1);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    reflect_factor = 0.1 + 0.3 * (1 - norm_fit);
    reflect_factor = repmat(reflect_factor, 1, D);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = lb_rep(below) + reflect_factor(below) .* (lb_rep(below) - offspring(below));
    offspring(above) = ub_rep(above) - reflect_factor(above) .* (offspring(above) - ub_rep(above));
    
    % 8. Final clamping with small perturbation
    offspring = max(min(offspring, ub_rep), lb_rep);
    offspring = offspring + 0.01 * (ub - lb) .* randn(NP,D) .* (1 - norm_fit(:,ones(1,D)));
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % 9. Elite preservation (top 10%)
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, round(0.1*NP));
    elite = popdecs(sorted_idx(1:elite_size),:);
    offspring(sorted_idx(1:elite_size),:) = elite + 0.005 * (ub - lb) .* randn(elite_size, D);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
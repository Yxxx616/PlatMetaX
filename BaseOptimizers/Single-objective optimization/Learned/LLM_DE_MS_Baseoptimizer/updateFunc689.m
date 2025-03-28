% MATLAB Code
function [offspring] = updateFunc689(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Feasibility-weighted elite selection
    sigma_c = std(cons);
    w = zeros(NP,1);
    feasible = cons <= 0;
    w(feasible) = 1./(1 + cons(feasible));
    w(~feasible) = exp(-cons(~feasible)/sigma_c);
    w = w/sum(w);
    elite_idx = randsample(NP, 1, true, w);
    elite = popdecs(elite_idx, :);
    
    % 2. Compute adaptive scaling factors
    sigma_f = std(popfits);
    F_base = 0.5;
    alpha = exp(-popfits/sigma_f);
    beta = 1 - alpha;
    F = F_base * (0.5 + 0.5*alpha);
    
    % 3. Compute hybrid direction vectors
    centroid = mean(popdecs, 1);
    elite_dir = bsxfun(@minus, elite, popdecs);
    centroid_dir = bsxfun(@minus, centroid, popdecs);
    rand_dir = randn(NP, D);
    
    % 4. Constraint-guided mutation
    lambda = 0.1;
    cons_factor = 1 + lambda*abs(cons);
    direction = alpha.*elite_dir + (1-alpha).*centroid_dir + beta.*rand_dir;
    mutant = popdecs + F.*direction.*cons_factor(:, ones(1,D));
    
    % 5. Rank-exponential crossover
    penalty = popfits + 1e6*max(0, cons);
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR_base = 0.9;
    gamma = 2;
    CR = CR_base * exp(-gamma*ranks/NP);
    
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Bounce-back boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = lb_rep(below_lb) + rand(sum(sum(below_lb)),1).*(popdecs(below_lb)-lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - rand(sum(sum(above_ub)),1).*(ub_rep(above_ub)-popdecs(above_ub));
end
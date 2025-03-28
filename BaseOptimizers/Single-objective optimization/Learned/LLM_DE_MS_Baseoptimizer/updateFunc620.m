% MATLAB Code
function [offspring] = updateFunc620(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Rank-based scaling factors
    [~, rank_idx] = sort(popfits);
    F = 0.5 + 0.3 * (rank_idx/NP); % Adaptive F
    
    % Constraint-aware perturbation
    cons_norm = abs(cons)/max(abs(cons)+eps);
    alpha = 0.2 * cons_norm;
    perturbation = alpha .* sign(cons) .* randn(NP, D);
    
    % Random indices selection
    idx = 1:NP;
    r1 = zeros(NP,1);
    r2 = zeros(NP,1);
    for i = 1:NP
        available = setdiff(idx, i);
        r1(i) = available(randi(length(available)));
        available = setdiff(available, r1(i));
        r2(i) = available(randi(length(available)));
    end
    
    % Mutation
    elite_rep = repmat(elite, NP, 1);
    diff = popdecs(r1,:) - popdecs(r2,:);
    F_rep = repmat(F, 1, D);
    mutant = elite_rep + F_rep.*diff + perturbation;
    
    % Rank-based crossover rates
    CR = 0.2 + 0.6 * (rank_idx/NP);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
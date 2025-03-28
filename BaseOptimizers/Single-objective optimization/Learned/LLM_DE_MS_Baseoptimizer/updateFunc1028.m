% MATLAB Code
function [offspring] = updateFunc1028(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + 1e-12);
    abs_cons = abs(cons);
    
    % Select elite (top 20%)
    [~, elite_idx] = sort(popfits);
    elite_num = max(1, floor(0.2*NP));
    elite = popdecs(elite_idx(1:elite_num),:);
    x_elite = mean(elite, 1);
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx,:);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx); r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx); r2 = r2 + (r2 >= idx);
    r3 = arrayfun(@(i) randi(NP-1), idx); r3 = r3 + (r3 >= idx);
    r4 = arrayfun(@(i) randi(NP-1), idx); r4 = r4 + (r4 >= idx);
    
    % Adaptive parameters
    F = 0.5 + 0.5 * (1 - norm_fits); % [0.5,1.0]
    eta = 0.1 * randn(NP, 1);
    sigma = 0.5 * (1 - norm_fits);
    w = norm_fits;
    
    % Constraint-aware perturbation weights
    cons_weights = 1./(1 + abs_cons(r1) + abs_cons(r2));
    
    % Mutation components
    dir_vectors = x_elite - popdecs;
    pert_vectors = (popdecs(r1,:) - popdecs(r2,:)) .* cons_weights;
    jump_vectors = x_best + sigma.*(popdecs(r3,:) - popdecs(r4,:));
    
    % Combined mutation
    mutants = popdecs + F.*dir_vectors + eta.*pert_vectors + (1-w).*jump_vectors;
    
    % Dynamic crossover rate
    CR = 0.9 - 0.5 * norm_fits;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Crossover
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
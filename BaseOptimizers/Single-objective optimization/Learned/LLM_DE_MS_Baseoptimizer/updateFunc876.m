% MATLAB Code
function [offspring] = updateFunc876(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(cons);
    alpha = 10;  % Constraint weight parameter
    
    % Precompute constraint weights for all possible triplets
    exp_cons = exp(-alpha * cons);
    
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select three distinct random individuals
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Compute constraint weights
        w = exp_cons([r1,r2,r3]) / sum(exp_cons([r1,r2,r3]));
        
        % Adaptive scaling factor
        F = 0.5 * (1 + (popfits(i) - f_min) / (f_max - f_min + 1e-12));
        
        % Mutation
        mutant = popdecs(r1,:) + F * (w(2)*popdecs(r2,:) - w(3)*popdecs(r3,:));
        
        % Adaptive crossover
        CR = 0.9 - 0.5 * (cons(i) / (c_max + 1e-12));
        mask = rand(1,D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        
        % Create offspring
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutant(mask);
    end
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Random perturbation for worst 10% solutions
    [~, sorted_cons] = sort(cons, 'descend');
    num_perturb = ceil(0.1*NP);
    perturb_idx = sorted_cons(1:num_perturb);
    for i = 1:num_perturb
        dim = randi(D);
        offspring(perturb_idx(i), dim) = lb(dim) + rand()*(ub(dim)-lb(dim));
    end
end
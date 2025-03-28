% MATLAB Code
function [offspring] = updateFunc155(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Weight calculation combining fitness and constraints
    median_fit = median(popfits);
    weights = 1./(1 + abs(popfits - median_fit) + abs(cons));
    
    % 2. Elite selection (top 30%)
    [~, sorted_idx] = sort(weights, 'descend');
    elite_idx = sorted_idx(1:ceil(0.3*NP));
    elite = popdecs(elite_idx, :);
    
    % 3. Adaptive parameters based on constraints
    max_cons = max(abs(cons)) + eps;
    F = 0.5 + 0.3 * cos(pi * abs(cons)/max_cons);
    CR = 0.8 - 0.4 * tanh(2 * abs(cons)/max_cons);
    
    % 4. Generate random indices
    rand_idx = randi(NP, NP, 4);
    r1 = rand_idx(:,1); r2 = rand_idx(:,2);
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % Select random elite for each individual
    elite_rand_idx = elite_idx(randi(length(elite_idx), NP, 1));
    
    % 5. Direction vectors with adaptive exploration-exploitation
    exploit_mask = rand(NP,1) < 0.6;
    d = zeros(NP, D);
    
    % Exploitation direction (DE/rand/2)
    d(exploit_mask,:) = popdecs(r1(exploit_mask),:) - popdecs(r2(exploit_mask),:) + ...
                        F(exploit_mask).*(popdecs(r3(exploit_mask),:) - popdecs(r4(exploit_mask),:));
    
    % Exploration direction (DE/current-to-elite/1)
    d(~exploit_mask,:) = popdecs(elite_rand_idx(~exploit_mask),:) - popdecs(~exploit_mask,:) + ...
                         F(~exploit_mask).*(popdecs(r1(~exploit_mask),:) - popdecs(r2(~exploit_mask),:));
    
    % 6. Mutation
    v = popdecs + F.*d;
    
    % 7. Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 8. Boundary handling with feasibility consideration
    feasible = cons <= median(cons);
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % For feasible solutions, reflect towards elite
    reflect_mask = repmat(feasible, 1, D) & ((offspring < lb_rep) | (offspring > ub_rep));
    offspring(reflect_mask) = 0.5*(offspring(reflect_mask) + popdecs(elite_rand_idx,:)(reflect_mask));
    
    % For infeasible solutions, random reinitialization
    infeasible_mask = repmat(~feasible, 1, D);
    reinit_mask = infeasible_mask & ((offspring < lb_rep) | (offspring > ub_rep));
    offspring(reinit_mask) = lb_rep(reinit_mask) + rand(sum(reinit_mask(:)),1).* ...
                            (ub_rep(reinit_mask) - lb_rep(reinit_mask));
    
    % Final projection for any remaining violations
    remaining = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(remaining) = max(min(offspring(remaining), ub_rep(remaining)), lb_rep(remaining));
end
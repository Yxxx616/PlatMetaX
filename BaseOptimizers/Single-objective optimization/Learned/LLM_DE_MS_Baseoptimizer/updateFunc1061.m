% MATLAB Code
function [offspring] = updateFunc1061(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Rank-based selection probabilities
    [~, sorted_idx] = sort(popfits, 'descend');
    ranks = zeros(NP,1);
    ranks(sorted_idx) = NP:-1:1;
    prob = (NP - ranks + 1) / sum(NP - ranks + 1);
    
    % 2. Select indices using rank probabilities
    idx = randsample(NP, NP*3, true, prob);
    r1 = idx(1:NP);
    r2 = idx(NP+1:2*NP);
    r3 = idx(2*NP+1:end);
    
    % 3. Constraint-aware adaptation factor
    max_cons = max(abs(cons)) + eps;
    alpha = 0.5 + 0.5 * tanh(abs(cons)/max_cons);
    
    % 4. Dynamic scaling factors
    F1 = 0.6 + 0.2 * alpha;
    F2 = 0.4 + 0.3 * (1 - alpha);
    
    % 5. Fitness-weighted differences
    fit_diff = abs(popfits(r1) - abs(popfits(r2));
    fit_weights = 1./(1 + exp(-fit_diff));
    weighted_diff = (popdecs(r1,:) - popdecs(r2,:)) .* fit_weights;
    
    % 6. Elite guidance (top 10%)
    elite_num = max(1, round(0.1*NP));
    elite_mean = mean(popdecs(sorted_idx(1:elite_num),:);
    
    % 7. Mutation strategy
    mutants = popdecs + ...
              F1.*(elite_mean - popdecs) + ...
              F2.*weighted_diff + ...
              0.1*(popdecs(r3,:) - popdecs);
    
    % 8. Dynamic crossover
    CR = 0.85 - 0.3*alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with random reinitialization
    out_of_bounds = offspring < lb | offspring > ub;
    rand_vals = lb + (ub-lb).*rand(NP,D);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end
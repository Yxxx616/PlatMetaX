% MATLAB Code
function [offspring] = updateFunc1584(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weighted centroid
    f_mean = mean(popfits);
    f_std = max(std(popfits), 1e-6);
    weights = 1 ./ (1 + exp((popfits - f_mean)/f_std));
    weights = weights ./ sum(weights);
    centroid = weights' * popdecs;
    
    % 2. Identify elite solutions (top 20%)
    cf = popfits + 1e6*max(0, cons);
    [~, sorted_idx] = sort(cf);
    elite_pool = popdecs(sorted_idx(1:ceil(NP*0.2)), :);
    x_elite = elite_pool(1, :); % best elite
    
    % 3. Compute personal best (top 30% solutions)
    pbest_pool = popdecs(sorted_idx(1:ceil(NP*0.3)), :);
    pbest = pbest_pool(randi(size(pbest_pool,1), NP, 1), :);
    
    % 4. Compute adaptive direction vectors
    dir_vectors = zeros(NP, D);
    alpha = 0.4 + 0.4*rand(NP,1);
    beta = 0.1 + 0.2*rand(NP,1);
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    
    feasible = cons <= 0;
    for i = 1:NP
        if feasible(i)
            dir_vectors(i,:) = x_elite - popdecs(i,:);
        else
            dir_vectors(i,:) = alpha(i)*(centroid - popdecs(i,:)) + ...
                              beta(i)*(popdecs(rand_idx1(i),:) - popdecs(rand_idx2(i),:));
        end
    end
    
    % 5. Compute adaptive scaling factors
    [~, ranks] = sort(popfits);
    F = 0.4 + 0.4*(1 - ranks/NP);
    
    % 6. Generate mutation vectors
    eta = 0.2 * rand(NP,1);
    offspring = popdecs + F.*dir_vectors + eta.*(pbest - popdecs);
    
    % 7. Rank-based crossover
    CR = 0.3 + 0.5*(1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % 9. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
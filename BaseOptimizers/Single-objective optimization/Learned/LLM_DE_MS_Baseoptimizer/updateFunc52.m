function [offspring] = updateFunc52(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps;
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    elite_group = sorted_idx(1:ceil(0.2*NP));
    rest_group = sorted_idx(ceil(0.2*NP)+1:end);
    
    for i = 1:NP
        % Select elite individual
        elite_idx = elite_group(randi(length(elite_group)));
        elite = popdecs(elite_idx, :);
        
        % Select 4 distinct random individuals from remaining population
        idxs = rest_group(randperm(length(rest_group), 4));
        x1 = popdecs(idxs(1), :);
        x2 = popdecs(idxs(2), :);
        x3 = popdecs(idxs(3), :);
        x4 = popdecs(idxs(4), :);
        
        % Get corresponding fitness values
        f1 = norm_fits(idxs(1));
        f2 = norm_fits(idxs(2));
        f3 = norm_fits(idxs(3));
        f4 = norm_fits(idxs(4));
        
        % Calculate weighted differences
        w1 = f2 / (f2 + f3 + eps);
        diff1 = w1 * (x2 - x3);
        
        w2 = f4 / (f4 + norm_fits(i) + eps);
        diff2 = w2 * (x4 - popdecs(i,:));
        
        % Adaptive F parameter based on constraint violation
        F = 0.5 * (1 + tanh(1 - norm_cons(i)));
        
        % Mutation
        v = elite + F * (diff1 + diff2);
        
        % Boundary control
        lb = -100 * ones(1, D);
        ub = 100 * ones(1, D);
        v = min(max(v, lb), ub);
        
        offspring(i,:) = v;
    end
end
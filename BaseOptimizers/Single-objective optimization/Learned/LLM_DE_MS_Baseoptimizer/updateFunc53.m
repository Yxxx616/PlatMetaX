function [offspring] = updateFunc53(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize fitness (convert to maximization problem)
    norm_fits = -popfits;
    norm_fits = (norm_fits - min(norm_fits)) / (max(norm_fits) - min(norm_fits) + eps);
    
    % Normalize constraint violations
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
        
        % Calculate weighted difference
        w1 = f2 / (f2 + f3 + eps);
        diff1 = w1 * (x2 - x3);
        
        % Adaptive F parameter based on constraint violation
        F = 0.5 + 0.5 * tanh(1 - norm_cons(i));
        
        % Mutation with opposition-based learning
        if rand() < 0.7
            v = elite + F * diff1 + 0.5*F*(elite - popdecs(i,:));
        else
            v = elite + F * diff1 + 0.5*F*(x4 - popdecs(i,:));
        end
        
        % Boundary control
        lb = -100 * ones(1, D);
        ub = 100 * ones(1, D);
        v = min(max(v, lb), ub);
        
        offspring(i,:) = v;
    end
end
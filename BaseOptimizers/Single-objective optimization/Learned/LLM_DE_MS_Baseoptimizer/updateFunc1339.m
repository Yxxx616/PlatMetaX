% MATLAB Code
function [offspring] = updateFunc1339(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    epsilon = 1e-12;
    
    % 1. Sort and partition population
    [~, sorted_idx] = sortrows([popfits, cons], [1 2]);
    pop_sorted = popdecs(sorted_idx, :);
    fits_sorted = popfits(sorted_idx);
    cons_sorted = cons(sorted_idx);
    
    elite_num = max(2, ceil(0.2*NP));
    middle_num = ceil(0.6*NP);
    elite = pop_sorted(1:elite_num, :);
    middle = pop_sorted(elite_num+1:elite_num+middle_num, :);
    inferior = pop_sorted(elite_num+middle_num+1:end, :);
    center = mean(elite, 1);
    x_best = pop_sorted(1,:);
    
    % 2. Calculate adaptive parameters
    c_max = max(abs(cons)) + epsilon;
    F_base = 0.5;
    F_rand = 0.3 * rand(NP, 1);
    F = F_base + F_rand .* (1 - abs(cons)./c_max);
    CR = 0.9 - 0.5 * (abs(cons)./c_max);
    
    % Pre-generate random indices for vectorization
    rand_idx1 = randi(NP, NP, 4);
    rand_idx2 = randi(NP, NP, 4);
    
    % 3. Generate offspring
    for i = 1:NP
        % Ensure indices are distinct and not equal to i
        idxs = mod(i + randperm(NP-1, 4), NP) + 1;
        
        % Determine mutation strategy based on rank
        if ismember(i, sorted_idx(1:elite_num))
            % Elite mutation
            mutant = x_best + F(i) .* (popdecs(idxs(1),:) - popdecs(idxs(2),:)) + ...
                     F(i) .* (popdecs(idxs(3),:) - popdecs(idxs(4),:));
        elseif ismember(i, sorted_idx(elite_num+1:elite_num+middle_num))
            % Middle mutation
            mutant = popdecs(i,:) + F(i) .* (popdecs(idxs(1),:) - popdecs(idxs(2),:)) + ...
                     F(i) .* (center - popdecs(i,:));
        else
            % Inferior mutation
            mutant = center + F(i) .* (popdecs(idxs(1),:) - popdecs(idxs(2),:)) + ...
                     F(i) .* (popdecs(idxs(3),:) - popdecs(idxs(4),:));
        end
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR(i);
        mask(j_rand) = true;
        trial = popdecs(i,:);
        trial(mask) = mutant(mask);
        
        % Constraint repair
        if cons(i) > 0
            beta = 0.8 * min(1, abs(cons(i))/c_max);
            elite_repair = elite(randi(elite_num), :);
            trial = (1-beta)*trial + beta*elite_repair;
        end
        
        offspring(i,:) = trial;
    end
    
    % Boundary handling with vectorization
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = lb(j) + rand(sum(below),1) .* (ub(j)-lb(j));
        offspring(above,j) = lb(j) + rand(sum(above),1) .* (ub(j)-lb(j));
    end
end
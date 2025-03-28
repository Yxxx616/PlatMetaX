classdef TSPs < PROBLEM
% <2007> <single> <permutation> <large/none>
% The traveling salesman problem

%------------------------------- Reference --------------------------------
% D. Corne and J. Knowles. Techniques for highly multiobjective
% optimisation: some nondominated points are better than others.
% Proceedings of the Annual Conference on Genetic and Evolutionary
% Computation, 2007, 773-780.
    properties(SetAccess = private)
        R;  % Locations of points
        C;  % Adjacency matrix
        instanceID;  % Unique ID for each instance
        instanceFeatures;  % Features of the instance (e.g., number of cities, distance statistics)
    end

    methods
        %% Default settings of the problem
        function Setting(obj)
            % Parameter setting
            obj.M = 1;
            if isempty(obj.D); obj.D = 30; end
            obj.encoding = 5 + zeros(1, obj.D);

            % Generate unique instance ID
            obj.instanceID = randi(1e6);  % Random ID for the instance

            % Randomly generate the adjacency matrix
            obj.R = rand(obj.D, 2);  % Random 2D coordinates for cities
            obj.C = pdist2(obj.R, obj.R);  % Distance matrix

            % Extract instance features
            obj.instanceFeatures = obj.extractFeatures();
        end
        
        function generateIns(obj, numCities)
            obj.D = numCities;  % Set number of cities
            obj.encoding = 5 + zeros(1, obj.D);

            % Generate unique instance ID
            obj.instanceID = randi(1e6);  % Random ID for the instance

            % Randomly generate the adjacency matrix
            obj.R = rand(obj.D, 2);  % Random 2D coordinates for cities
            obj.C = pdist2(obj.R, obj.R);  % Distance matrix

            % Extract instance features
            obj.instanceFeatures = obj.extractFeatures();
        end

        %% Extract features from the TSP instance
        function ff = extractFeatures(obj) %17
            % 1. Basic features
            features.numCities = obj.D;  % Number of cities

            % 2. Distance matrix features
            distances = obj.C(:);  % Flatten the distance matrix
            features.meanDistance = mean(distances);
            features.stdDistance = std(distances);
            features.minDistance = min(distances);
            features.maxDistance = max(distances);
            features.medianDistance = median(distances);

            % 3. Graph features
            % Compute the degree of each city (number of connections)
            degrees = sum(obj.C > 0, 2);  % Degree of each city
            features.meanDegree = mean(degrees);
            features.stdDegree = std(degrees);

            % Compute the clustering coefficient (optional, requires graph toolbox)
            try
                G = graph(obj.C);
                clusteringCoeffs = clustering_coefficient(G);
                features.meanClusteringCoeff = mean(clusteringCoeffs);
            catch
                features.meanClusteringCoeff = NaN;  % If graph toolbox is not available
            end

            % 4. Geometric features
            % Compute the convex hull area
            try
                k = convhull(obj.R(:, 1), obj.R(:, 2));
                hullArea = polyarea(obj.R(k, 1), obj.R(k, 2));
                features.convexHullArea = hullArea;
            catch
                features.convexHullArea = NaN;  % If convhull fails
            end

            % Compute the centroid of the cities
            centroid = mean(obj.R, 1);
            features.centroidX = centroid(1);
            features.centroidY = centroid(2);

            % Compute the spread of cities (standard deviation of coordinates)
            features.stdX = std(obj.R(:, 1));
            features.stdY = std(obj.R(:, 2));

            % 5. Additional features (optional)
            % Ratio of convex hull area to total area
            totalArea = range(obj.R(:, 1)) * range(obj.R(:, 2));
            features.hullAreaRatio = features.convexHullArea / totalArea;

            % Skewness and kurtosis of distances
            features.skewnessDistance = skewness(distances);
            features.kurtosisDistance = kurtosis(distances);
            ff = struct2array(features);
        end

        %% Record performance data for a meta-heuristic
        function recordPerformance(obj, algorithmName, solutionQuality, computationTime)
            % Store performance data for the given algorithm
            obj.performanceData.(algorithmName) = struct(...
                'solutionQuality', solutionQuality, ...
                'computationTime', computationTime);
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            [sorted,rank] = sort(PopDec,2);
            index = any(sorted~=repmat(1:size(PopDec,2),size(PopDec,1),1),2);
            PopDec(index,:) = rank(index,:);
            PopObj = zeros(size(PopDec,1),1);
            for i = 1 : size(PopDec,1)
                for j = 1 : size(PopDec,2)-1
                    PopObj(i) = PopObj(i) + obj.C(PopDec(i,j),PopDec(i,j+1));
                end
                PopObj(i) = PopObj(i) + obj.C(PopDec(i,end),PopDec(i,1));
            end
        end
        
        %% Display a population in the decision space
        function DrawDec(obj, Population)
            [~, best] = min(Population.objs);
            if any(~ismember(1:length(Population(best).dec), Population(best).dec))
                [~, Dec] = sort(Population(best).dec);
            else
                Dec = Population(best).dec;
            end
            Draw(obj.R(Dec([1:end, 1]), :), '-k', 'LineWidth', 1.5);
            Draw(obj.R);
        end
        
        function saveInstance(obj, filename)
            % Save instance data for later use
            save(filename, 'obj');
        end
    end
end
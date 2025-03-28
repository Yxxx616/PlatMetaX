classdef DDPG_DE_Alpha_Metaoptimizer < rl.agent.AbstractContinuousActionAgentMemoryTarget
    properties (Access = private)
        % Critic function approximator
        Critic
        
        % Target critic function approximator
        TargetCritic
        
        % Noise model
        NoiseModel
    end
    
    properties (Access = private)
        % version indicator for backward compatibility
        Version = 2
    end

    methods
        function this = DDPG_DE_Alpha_Metaoptimizer(obsInfo, actInfo)
            % Constructor
            AgentType = "DDPG";
            [ObservationInfo,ActionInfo,InitOptions,AgentOptions] = rl.util.parseAgentInitializationInputs(AgentType,obsInfo,actInfo);
            Actor = rl.representation.rlDeterministicActorRepresentation.createDefault(ObservationInfo, ActionInfo, InitOptions);
            Critic = rl.representation.rlQValueRepresentation.createDefault(ObservationInfo, ActionInfo, InitOptions, 'singleOutput');
       
            rl.util.validateAgentOptionType(AgentOptions,AgentType);
            
            this = this@rl.agent.AbstractContinuousActionAgentMemoryTarget(...
                Actor.ObservationInfo,Actor.ActionInfo);
            
			rl.agent.AbstractContinuousActionAgentMemoryTarget.validateActionInfo(Actor.ActionInfo); 
            
            % set agent option
            this.AgentOptions = AgentOptions;
            
            % set representations
            setActor (this, Actor );
            setCritic(this, Critic);
            this.HasCritic = true;
        end
        
        %==================================================================
        % Get/set
        %==================================================================
        function critic = getCritic(this)
            % getCritic: Return the critic representation, CRITIC, for the
            % specified reinforcement learning agent, AGENT.
            %
            %   CRITIC = getCritic(AGENT)
            
            critic = this.Critic;
        end
        function this = setCritic(this, Critic)
            % setCritic: Set the critic of the reinforcement learning agent
            % using the specified representation, CRITIC, which must be
            % consistent with the observations and actions of the agent.
            %
            %   AGENT = setCritic(AGENT,CRITIC)
            
            % validate critic is a single output Q representation
            validateattributes(Critic, {'rl.representation.rlQValueRepresentation'}, {'scalar', 'nonempty'}, '', 'Critic');
            
            % validate critic is single output Q representations
            if strcmpi(getQType(Critic),'multiOutput')
                error(message('rl:agent:errDPGMultiQ'))
            end
            
            % check if actor and critic created from same data specs and if
            % both stateless or both have state
            rl.agent.AbstractAgent.validateActorCriticInfo(Critic,this.Actor)
            
            % validate against agent options
            rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(this.AgentOptions, Critic);
            
            % set critic loss
            Critic = setLoss(Critic,"rl.loss.rlmse");
            
            % Set critic network
            this.Critic = Critic;
            
            % Construct target network
            this.TargetCritic = this.Critic;
            
            reset(this);
        end        
    end 
    
    %======================================================================
    % Implementation of abstract methods
    %======================================================================
    methods (Access = protected)
        function options = setAgentOptionsImpl(this,options)
            rl.util.validateAgentOptionType(options,'DDPG');
            rebuildNoise = isempty(this.NoiseModel) || ...
                ~isequal(this.AgentOptions_.NoiseOptions,options.NoiseOptions);
            % build the noise model if necessary
            if rebuildNoise
                % extract the noise options
                noiseOpts = options.NoiseOptions;

                % create the noise model
                actionDims = {this.ActionInfo.Dimension}';
                this.NoiseModel = rl.util.createNoiseModelFactory(...
                    actionDims,noiseOpts,options.SampleTime);
            end
            
            rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(options, this.Critic);
            rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(options, this.Actor);
        end
        function varargout = stepRepresentation(this,minibatch,maskIdx)
            % Update the critic, actor, target critic and target actor
            % given a minibatch and a target discount factor
            
            if nargout
                % This branch supports gorilla parallel training where
                % workers send gradients to learners. 'train' methods
                % output a struct that contains critic and actor gradients.
                % The gradient will be supply to applyGradient(this,grad)
                s.Critic = trainCriticWithBatch(this,minibatch,maskIdx);
                s.Actor  = trainActorWithBatch (this,minibatch,maskIdx);
                varargout{1} = s;
            else
                % update the critic
                trainCriticWithBatch(this,minibatch,maskIdx);
                % update the actor
                trainActorWithBatch(this,minibatch,maskIdx);
                % update the target representations
                updateTargetRepresentations(this);
            end
        end
        function resetImpl(this)
            % rebuild agent properties due to any potential changes in
            % options
            
            % reset the exp buffer
            resetImpl@rl.agent.AbstractContinuousActionAgentMemoryTarget(this);
            
            % reset the noise model
            reset(this.NoiseModel);
            
            % reset state representation
            if ~isempty(this.Critic)
                this.Critic = resetState(this.Critic);
                this.TargetCritic = resetState(this.TargetCritic);
            end
            if ~isempty(this.Actor)
                this.Actor = resetState(this.Actor);
                this.TargetActor = resetState(this.TargetActor);
            end
        end
        
        function q0 = evaluateQ0Impl(this,observation)
            % get the estimated long-term return (Q0)
            
            % reset representation state (RNN), no-op for non RNN
            this.Actor = resetState(this.Actor);
            this.Critic = resetState(this.Critic);
            
            % Q(s0,mu(s0))
            action = getAction(this.Actor, observation);
            q0 = getValue(this.Critic, observation, action);
            if isa(q0,'dlarray')
                q0 = extractdata(q0);
            end
        end
        
        function noiseMdl = getNoiseModelForTraining(this)
            noiseMdl = this.NoiseModel;
        end
        
        % set/get tunable parameters
        function setLearnableParametersImpl(this,p)
            this.Actor  = setLearnableParameters(this.Actor ,p.Actor );
            this.Critic = setLearnableParameters(this.Critic,p.Critic);
        end
        
        function p = getLearnableParametersImpl(this)
            p.Actor  = getLearnableParameters(this.Actor );
            p.Critic = getLearnableParameters(this.Critic);
        end
        
        function that = copyElement(this)
            that = copyElement@rl.agent.AbstractContinuousActionAgentMemoryTarget(this);
            that.NoiseModel = copy(this.NoiseModel);
        end
    end
    
    methods (Hidden)
        function applyGradient(this,g)
            % update representation from gradients
            if ~isempty(g)
                this.Critic = optimize(this.Critic,g.Critic);
                this.Actor = optimize(this.Actor ,g.Actor);
                updateTargetRepresentations(this);
            end
        end        
    end
    
    %======================================================================
    % Step representation methods
    %======================================================================
    methods (Access = private)
        function updateTargetRepresentations(this)
            % Update the target networks
            this.TargetCritic = updateTargetFromOpts(this,this.Critic,this.TargetCritic);
            this.TargetActor  = updateTargetFromOpts(this,this.Actor ,this.TargetActor );
        end
        
        function varargout = trainCriticWithBatch(this,miniBatch,maskIdx)
            % update the critic against a minibatch set
            
            observations     = miniBatch{1};
            actions          = miniBatch{2};
            rewards          = miniBatch{3};
            nextobservations = miniBatch{4};
            isdones          = miniBatch{5};
            
            % reset representation state (RNN), no-op for non RNN
            this.Critic = resetState(this.Critic);
            
            % compute the next actions from the target actor
            nextactions = getAction(this.TargetActor,nextobservations);
            
            % compute the next step expected Q value (bootstrapping)
            targetq = getValue(this.TargetCritic,nextobservations,nextactions);
            doneidx = isdones == 1;
            gamma = this.AgentOptions.DiscountFactor;
            n = this.AgentOptions.NumStepsToLookAhead;
            
            % get target Q values we should expect the network to work
            % towards
            targetq(~doneidx) = rewards(~doneidx) + (gamma^n).*targetq(~doneidx);
            
            % for final step, just use the immediate reward, since there is
            % no more a next state
            targetq(doneidx) = rewards(doneidx);
            
            % dummification of q target from RNN patching
            if hasState(this)
                % bypass of data is not patched
                if ~all(maskIdx,'all')
                    qPrediction = getValue(this.Critic,observations,actions);
                    targetq(~maskIdx) = qPrediction(~maskIdx);
                end
            end
            
            % train the critic or get the gradients
            criticGradient = gradient(this.Critic,'loss-parameters',...
                    [observations, actions], targetq);
            if nargout
                varargout{1} = criticGradient;
            else
                this.Critic = optimize(this.Critic, criticGradient);
            end
        end
        
        function varargout = trainActorWithBatch(this,miniBatch,maskIdx)
            % update the actor against a minibatch set
            
            if hasState(this.Actor)
                    % save the current state of actor (apply next env step)
                    currentState = getState(this.Actor);
                    % reset actor and critic state before learning
                    this.Actor  = resetState(this.Actor);
                    this.Critic = resetState(this.Critic);
                    % numExperience == number of non patched experience
                    numExperience = sum(maskIdx,'all');
                else
                    % numExperience == number of rewards
                    numExperience = numel(miniBatch{3});
            end

            if rl.util.rlfeature("UseDlnetwork")
                lossVariable.Observation = miniBatch{1};
                lossVariable.Critic = this.Critic;
                lossVariable.MaskIdx = maskIdx;
                lossVariable.NumObs = numExperience;
                % pass observation and critic to 'rl.loss.dpg' loss
                actorGradient = gradient(this.Actor,'loss-parameters',lossVariable.Observation,lossVariable);
            else
                observations = miniBatch{1};
            
                % get the actions
                actions = getAction(this.Actor,observations);
                
                % get dQ/da
                if hasState(this)
                    dQdInput = gradient(this.Critic,'output-input',[observations,actions],{maskIdx});
                else
                    dQdInput = gradient(this.Critic,'output-input',[observations,actions]);
                end
                dQdA = dQdInput{end};
                
                % get da/dParams
                % for gradient ascent we invert the sign of the initial
                % gradients from the critic.
                actorGradient = gradient(this.Actor,'output-parameter',observations,{-dQdA});
                
                % divide the gradients by numObservations and apply them to the
                % actor using the optimizer to smooth the gradients.
                actorGradient = rl.internal.dataTransformation.scaleLearnables(actorGradient, 1/numExperience);
            end
            
            if nargout
                varargout{1} = actorGradient;
            else
                this.Actor = optimize(this.Actor, actorGradient);
            end
            
            if hasState(this.Actor)
                % feed RNN state for next env step
                this.Actor = setState(this.Actor,currentState);
            end
        end
    end
        
    methods (Static)
        function obj = loadobj(s)
            if isstruct(s)
                if ~isfield(s,'Version')
                    % version 1 (19a, 19b) and 2 (20a) not have Version field
                    % From version 2,
                    %   - Critic changes from rlRepresentation to rlQValueRepresentation
                    %   - Actor changes from rlRepresentation to rlDeterministicActorRepresentation
                    %   - ExperienceBuffer requires obs and act dims inputs
                    
                    if isa(s.Critic,'rl.util.rlAbstractRepresentation')
                        % version 1: reconstruct representations to new
                        % class rl.util.rlAbstractRepresentation
                        
                        criticModel = s.Critic.getModel;
                        targetCriticModel = s.TargetCritic.getModel;
                        if isa(criticModel,'DAGNetwork') || isa(criticModel,'nnet.cnn.LayerGraph')
                            s.Critic = rlQValueRepresentation(criticModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                'Observation', s.Critic.ObservationNames, ...
                                'Action', s.Critic.ActionNames, ...
                                s.Critic.Options);
                            s.TargetCritic = rlQValueRepresentation(targetCriticModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                'Observation', s.TargetCritic.ObservationNames, ...
                                'Action', s.TargetCritic.ActionNames, ...
                                s.TargetCritic.Options);
                        else
                            s.Critic = rlQValueRepresentation(criticModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                s.Critic.Options);
                            s.TargetCritic = rlQValueRepresentation(targetCriticModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                s.TargetCritic.Options);
                        end
                        
                        actorModel = s.Actor.getModel;
                        targetActorModel = s.TargetActor.getModel;
                        if isa(actorModel,'DAGNetwork') || isa(actorModel,'nnet.cnn.LayerGraph')
                            s.Actor = rlDeterministicActorRepresentation(actorModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                'Observation', s.Actor.ObservationNames, ...
                                'Action', s.Actor.ActionNames, ...
                                s.Actor.Options);
                            s.TargetActor = rlDeterministicActorRepresentation(targetActorModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                'Observation', s.TargetActor.ObservationNames, ...
                                'Action', s.TargetActor.ActionNames, ...
                                s.TargetActor.Options);
                        else
                            s.Actor = rlDeterministicActorRepresentation(actorModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                s.Actor.Options);
                            s.TargetActor = rlDeterministicActorRepresentation(targetActorModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                s.TargetActor.Options);
                        end
                    end
                end
                
                obj = rl.agent.rlDDPGAgent(s.Actor,s.Critic,s.AgentOptions_);
                obj.NoiseModel = s.NoiseModel;
                obj.TargetActor = s.TargetActor;
                obj.TargetCritic = s.TargetCritic;
                
                if obj.AgentOptions_.SaveExperienceBufferWithAgent
                    % only load the experience buffer if
                    % SaveExperienceBufferWithAgent is true
                    obj.ExperienceBuffer = s.ExperienceBuffer;
                end
            else
                obj = s;
            end
        end
    end
end

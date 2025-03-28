classdef NN_DE_OFF_Metaoptimizer < rl.agent.AbstractAgentMemoryTarget
    properties (Access = private)
        % Critic function approximator
        Critic
        
        % Target critic function approximator
        TargetCritic
        
        % Epsilon-Greedy Exploration parameters
        ExplorationModel
    end
    
    properties (Access = private)
        % version indicator for backward compatibility
        Version = 2
    end

    methods
        function this = NN_DE_OFF_Metaoptimizer(obsInfo, actInfo)
            AgentType = "DQN";
            [ObservationInfo,ActionInfo,InitOptions,AgentOptions] = rl.util.parseAgentInitializationInputs(AgentType,obsInfo,actInfo);
            Critic = rl.representation.rlQValueRepresentation.createDefault(ObservationInfo, ActionInfo, InitOptions, 'multiOutput');
   
            rl.util.validateAgentOptionType(AgentOptions,AgentType);
            this = this@rl.agent.AbstractAgentMemoryTarget(...
                Critic.ObservationInfo,Critic.ActionInfo);
			rl.agent.rlDQNAgent.validateActionInfo(Critic.ActionInfo);
            
            % set agent option
            this.AgentOptions = AgentOptions;

            % set representation
            setCritic(this,Critic);
            this.HasCritic = true;
        end
        
        function Action = getActionWithExploration(this,Observation) 
            % Return an action with exploration
            
            if rand < this.ExplorationModel.Epsilon
                Action = usample(this.ActionInfo);  
                if hasState(this)
                    % To update a hidden state of LSTM, we first get the
                    % current hidden state (output of getValue).
                    [~, State] = getValue(this.Critic, Observation);

                    % We update the hidden state using the current
                    % hidden state.
                    this.Critic = setState(this.Critic,State);
                end
            else
                % Hidden state is update in getActionImp
                Action = getAction(this,Observation); 
            end
           
            if strcmp(getStepMode(this),"sim-with-exploration")
                % (for parallel training) update the noise model on workers
                this.ExplorationModel = update(this.ExplorationModel);
            end
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
                        
            % validate critic is a Q representation
            validateattributes(Critic, {'rl.representation.rlQValueRepresentation'}, {'scalar', 'nonempty'}, '', 'Critic');
            
            % validate agent options
            validateOptionImpl(this, this.AgentOptions, Critic)
            
            % validate action and observation infos are same
            if ~isCompatible(this.ActionInfo,Critic.ActionInfo)
                error(message('rl:agent:errIncompatibleActionInfo'))
            end
            if ~isCompatible(this.ObservationInfo,Critic.ObservationInfo)
                error(message('rl:agent:errIncompatibleObservationInfo'))
            end
            
            if hasState(Critic) && (getQType(Critic) == "singleOutput")
                error(message('rl:agent:errStatefulSingleOutQDiscreteAction'));
            end
            
            % set loss
            if hasState(Critic)
                % loss function for DRQN (DQN that uses LSTM).
                Critic = setLoss(Critic,"rl.loss.drq");
            else
                % loss function for DQN that not use RNN.
                Critic = setLoss(Critic,"rl.loss.dq");
            end
            
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
    methods(Access = protected)
        function options = setAgentOptionsImpl(this,options)
            rl.util.validateAgentOptionType(options,'DQN');
            validateOptionImpl(this,options,this.Critic);
            % grab the exploration model
            this.ExplorationModel = options.EpsilonGreedyExploration;
        end
        function [rep,argStruct] = generateProcessFunctions_(this,argStruct)
            % DQN code gen
            rep = this.Critic;
            
            if strcmpi(getQType(this.Critic), 'multiOutput')
                argStruct = rl.codegen.generateDiscreteCriticMultiOutQFcn(argStruct,this.ActionInfo);
            else
                argStruct = rl.codegen.generateDiscreteCriticSingleOutQFcn(argStruct,this.ActionInfo);
            end
        end
        
        function q0 = evaluateQ0Impl(this,observation)
            % overload for agents that implement critics
            
            % RNN reset hidden state, no-op otherwise
            this.Critic = resetState(this.Critic);
            q0 = getMaxQValue(this.Critic, observation);
            if isa(q0,'dlarray')
                q0 = extractdata(q0);
            end
        end
        
        % set/get tunable parameters
        function setLearnableParametersImpl(this,p)
            this.Critic = setLearnableParameters(this.Critic,p.Critic);
        end
        function p = getLearnableParametersImpl(this)
            p.Critic = getLearnableParameters(this.Critic);
        end
        
        function Action = getActionImpl(this,Observation) 
            % Given the current state of the system, return an action.

            [Action,State] = getAction(this.Critic,Observation);
            this.Critic = setState(this.Critic, State);
        end
        
        function resetImpl(this)
            % reset the experience buffer
            resetImpl@rl.agent.AbstractAgentMemoryTarget(this);
            
            % Revert exploration model to original parameters
            this.ExplorationModel = this.AgentOptions.EpsilonGreedyExploration;
            this.Critic = resetState(this.Critic);
        end
        
        function varargout = learn(this,exp)
            % learn from the current set of experiences where
            % exp = {state,action,reward,nextstate,isdone}
            % Return the action with exploration.
            % NOTE: only learn, update params and exploration once exp
            % buffer has more experiences than MiniBatchSize

            % store experiences
            appendExperience(this,exp);
            
            % sample data from exp buffer and compute gradients
            Grad = accumulateGradient(this);
            
            % update critic params and update exploration
            applyGradient(this,Grad);
            
            if nargout
                % compute action from the current policy
                % {state,action,reward,nextstate,isdone}
                varargout{1} = getActionWithExploration(this,exp{4});
            end
        end
        
        function HasState = hasStateImpl(this)
            % whether use RNN
            HasState = hasState(this.Critic);
        end
        function resetStateImpl(this)
            % reset state of RNN policy, no-op for non-RNN
            this.Critic = resetState(this.Critic);
        end
        
        %% Experience based parallel methods ==============================
        % Methods to support experience based parallel training
        % =================================================================
        
        % Construct an Epsilon-greedy noisy policy for parallel training
        function policy = getParallelWorkerPolicyForTrainingImpl(this)
            % return a noisy policy for parallel training
            noiseMdl = this.ExplorationModel;
            % build the policy
            critic = this.Critic;
            % force the critic to use cpu as the workers will just be
            % executing 1 step predictions for simulation
            critic.Options.UseDevice = "cpu";
            policy = rl.policy.EpsilonDiscreteActionNoisyPolicy(...
                critic,noiseMdl,getSampleTime(this));
        end
        
        % Return the parameters that will be sent to the policies
        % simulating on the workers. DQN will return the critic parameters.
        function p = getParallelWorkerPolicyParametersImpl(this)
            p = getLearnableParameters(this.Critic);
            % make sure to convert gpu params to cpu params
            p = dlupdate(@gather,p);
        end
    end
    
    %======================================================================
    % Step representation methods
    %======================================================================
    methods (Access = private)
        function CriticGradient = trainCriticWithBatch(this,MiniBatch, MaskIdx)
            % update the critic against a minibatch set
            % MaskIdx is for DRQN. MaskIdx indicates actual inputs (true)
            % and padded inputs (false). MaskIdx is empty for DQN.         
            % MaskIdx is a tensor of size [1 x MiniBatchSize x SequenceLength].
            
            if this.AgentOptions.UseDoubleDQN
                % DoubleDQN: r + DiscountFactor*Q[s',a' = argmax(qNetwork(s'))]
                TargetCriticLocal = this.Critic;
            else
                % DQN:       r + DiscountFactor*Qtarget[s',a' = argmax(targetNetwork(s'))]
                TargetCriticLocal = this.TargetCritic;
            end
            
            % unpack experience
            Reward          = MiniBatch{3};
            NextObservation = MiniBatch{4};
            DoneIdx         = MiniBatch{5} == 1;
            Discount        = this.AgentOptions.DiscountFactor ^ ...
                                this.AgentOptions.NumStepsToLookAhead;
            
            % reset hidden state if network is RNN (LSTM)
            if hasState(this)
                TargetCriticLocal = resetState(TargetCriticLocal);
            end
            
            % compute Target Q values outside of the loss function
            TargetQValues = getMaxQValue(TargetCriticLocal, NextObservation);
            TargetQValues(~DoneIdx) = Reward(~DoneIdx) + ...
                Discount.*TargetQValues(~DoneIdx);

            % for terminal step, use the immediate reward (no more next state)
            TargetQValues(DoneIdx) = Reward(DoneIdx);
            
            % build loss variable struct
            LossVariable.Action        = MiniBatch{2};
            LossVariable.ActionInfo    = this.ActionInfo;
            LossVariable.QType         = getQType(TargetCriticLocal);
            LossVariable.UseDevice     = TargetCriticLocal.Options.UseDevice;
            LossVariable.TargetQValues = TargetQValues;
            
            if hasState(this)
                % RNN uses a mask to ignore padded inputs.
                LossVariable.MaskIdx = MaskIdx;
                % save the current hidden state to use at the next agent
                % step
                CurrentState = getState(this.Critic); 
                % reset hidden state before updating
                this.Critic = resetState(this.Critic);
            end
            
            % compute gradient
            % MiniBatch{1} is observation
            if strcmpi(getQType(this.Critic),'singleOutput')
                CriticGradient = gradient(this.Critic,'loss-parameters',...
                    [MiniBatch{1},LossVariable.Action], LossVariable);
            else
                CriticGradient = gradient(this.Critic,'loss-parameters',...
                MiniBatch{1}, LossVariable);
            end
            
            if hasState(this)
                % after updating, recover the state using the saved hidden state
                this.Critic = setState(this.Critic, CurrentState);
            end
        end
        
        function updateTargetRepresentations(this)
            % Update target critic parameters
            this.TargetCritic = updateTargetFromOpts(this,this.Critic,this.TargetCritic);
        end
        
        function validateOptionImpl(this,NewOptions,Critic)
            % Check compatibility of options and critic representation

            if ~isempty(Critic)
                if ~isempty(this.Critic)
                    if hasState(this.Critic) && ~hasState(Critic)
                        % If the original critic network architecture has state
                        % (e.g. LSTM, DRQN algorithm)), but new critic
                        % architecture does not have states (DQN algorithm), it
                        % is necessary to create a new agent with
                        % sequenceLength=1
                        error(message('rl:agent:errCriticChangedtoStateless'))

                    elseif ~hasState(this.Critic) && hasState(Critic)
                        % If the original critic network architecture does
                        % not have states(DQN algorithm), but new critic
                        % architecture has states (e.g. LSTM, DRQN
                        % algorithm), it is necessary to create a new agent
                        % with sequenceLength>1
                        error(message('rl:agent:errCriticChangedtoRNN'))
                    end
                end
                
                rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(NewOptions, Critic);
            end
        end
    end
    
    methods (Hidden)
        function Grad = accumulateGradient(this,~)
            % Accumulate representation gradients from current experiences,
            % return gradient values and parameter values
            
            % generate a minibatch: MiniBatchSize length of cell array with
            % {state,action,reward,nextstate,isdone} elements
                        
            MaskIdx = {}; %DQN does not use masking
            if hasState(this)
                [Minibatch, MaskIdx] = ...
                    createSampledExperienceMiniBatchSequence(this);
            else
                Minibatch = createSampledExperienceMiniBatch(this);                
            end
                                   
            if isempty(Minibatch)
                Grad = [];
            else
                % perform the learning on the representation
                Grad.Critic = trainCriticWithBatch(this, Minibatch, MaskIdx);
                
                % update exploration
                this.ExplorationModel = update(this.ExplorationModel);
            end
        end
        
        function applyGradient(this,Grad)
            % Update representation from gradients
            
            if ~isempty(Grad)
                this.Critic = optimize(this.Critic, Grad.Critic);
                updateTargetRepresentations(this);
            end
        end
        
        function actor = getActor(this) %#ok<MANU>
            % getActor: DRQN agent does not have actor. Therefore, it
            % returns empty.
            
            actor = [];
        end
        
        function policy = getSimulationPolicy(this)
            % return a representation policy of the discrete critic
            policy = rl.policy.RepresentationPolicy(this.Critic,getSampleTime(this));
        end
    end
    
    methods (Static)
        function obj = loadobj(s)
            if isstruct(s)
                if ~isfield(s,'Version')
                    % version 1 (19a, 19b) and 2 (20a) not have Version field
                    % From version 2,
                    %   - Critic changes from rlRepresentation to rlQValueRepresentation
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
                    end
                end

                obj = rl.agent.rlDQNAgent(s.Critic,s.AgentOptions_);
                obj.ExplorationModel = s.ExplorationModel;
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
    
    methods (Static,Hidden)
        function validateActionInfo(ActionInfo)
            
            if ~isa(ActionInfo,'rl.util.RLDataSpec')
                error(message('rl:agent:errInvalidActionSpecClass'))
            end
            
            % DQN does not support continuous action data spec
            if rl.util.isaSpecType(ActionInfo, 'continuous')
                error(message('rl:agent:errDQNContinuousActionSpec'))
            end
            
            % DQN does not support actions that are matrix or tensor since
            % getElementIndicationMatrix in rlFiniteSetSpec does not
            % support them. It supports only scalar and vector.
            if iscell(ActionInfo.Elements)
                OneAction = ActionInfo.Elements{1};
            else
                OneAction = ActionInfo.Elements(1);
            end
            if ~isscalar(OneAction) && ~isvector(OneAction)
                error(message('rl:agent:errDQNNotSupportMatrixTensorAction'))
            end
        end
    end
end
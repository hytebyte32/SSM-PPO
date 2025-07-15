import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


### learnable sigmoid function. Directly affects gating fuctionality
class ShiftedSigmoid(nn.Module):
    def __init__(self):
        super(ShiftedSigmoid, self).__init__()

        self.shift = T.pi

        # learns how smooth weights ramp up
        self.smoothness = nn.Parameter(T.tensor(1.5))

    def forward(self, x):
        y = nn.Sigmoid()(self.smoothness * x - self.shift)
        return y

class RecurrentBlock(nn.Module):
    def __init__(
            self,
            embedding_size,
            state_space_size,
            input_size,
            output_size,
            batch_size,
            device
            ):
        super(RecurrentBlock, self).__init__()

        self.device = device

        self.embedding_size = embedding_size
        self.state_space_size = state_space_size
        self.batch_size = batch_size
        
        ### learnable magnitude and phase parameters for state transition matrix
        fn_transition = lambda w: T.log(-T.log(w))
        self.real_transition_matrix = nn.Parameter(self.initialize_weights(low=0.999, high=1, dims=(embedding_size, state_space_size), fn=fn_transition))
        self.img_transition_matrix = nn.Parameter(self.initialize_weights(low=0, high=np.pi/10, dims=(embedding_size, state_space_size)))

        ### learnable magnitude and phase parameters for initial hidden state   
        fn_hidden = lambda w: T.log(-T.log(w))
        self.real_hidden_matrix = nn.Parameter(self.initialize_weights(low=0.001, high=0.002, dims=(embedding_size, state_space_size), fn=fn_hidden))
        self.img_hidden_matrix = nn.Parameter(self.initialize_weights(low=0, high=np.pi/10, dims=(embedding_size, state_space_size)))

        # constructing state transition matrix
        self.state_transition_matrix = self.construct_complex_matrix(
            real_values=self.real_transition_matrix,
            img_values=self.img_transition_matrix
        )
        
        # constructing initial hidden state
        self.hidden_state_matrix = self.construct_complex_matrix(
            real_values=self.real_hidden_matrix,
            img_values=self.img_hidden_matrix,
            #duplicate_batchwise=True
        )

        ### weight 
        # input gate       
        fn_inputs = lambda w:T.sqrt(1-w)
        input_gate_weights_real = self.initialize_weights(low=0.999, high=1, dims=(embedding_size, embedding_size), fn=fn_inputs)
        input_gate_weights_img = self.initialize_weights(low=0, high=np.pi/10, dims=(embedding_size, embedding_size))
        input_gate_weights = input_gate_weights_real * T.exp(input_gate_weights_img * 1j)

        # output gate
        fn_outputs = lambda w:T.sqrt(1-w)
        output_gate_weights_real = self.initialize_weights(low=0.999, high=1, dims=(embedding_size, embedding_size), fn=fn_outputs)
        output_gate_weights_img = self.initialize_weights(low=0, high=np.pi/10, dims=(embedding_size, embedding_size))
        output_gate_weights = output_gate_weights_real * T.exp(output_gate_weights_img * 1j)


        # discretization matrix
        fn_discretization = lambda w:T.sqrt(1-w)
        discretization_real = self.initialize_weights(low=0.001, high=0.002, dims=(embedding_size, embedding_size), fn=fn_discretization)
        discretization_img = self.initialize_weights(low=0, high=np.pi/10, dims=(embedding_size, embedding_size))
        discretization_weights = discretization_real * T.exp(discretization_img * 1j)

        ### recurrent block layers
        self.embedding_layer = nn.Linear(input_size, embedding_size, bias=False, dtype=T.complex64)
        discretization_matrix = nn.Linear(embedding_size, embedding_size, bias=False, dtype=T.complex64)
        input_gate = nn.Linear(embedding_size, embedding_size, bias=False, dtype=T.complex64)
        self.output_gate = nn.Linear(embedding_size, embedding_size, bias=False, dtype=T.complex64)

        # assigning weights
        with T.no_grad():
            input_gate.weight.copy_(input_gate_weights)
            discretization_matrix.weight.copy_(discretization_weights)
            self.output_gate.weight.copy_(output_gate_weights)


        # adding shifted sigmoid activation to gates
        self.input_gate = nn.Sequential(
            input_gate,
            ShiftedSigmoid()
            )

        self.discretization_matrix = nn.Sequential(
            discretization_matrix,
            ShiftedSigmoid()
            )
        
        # input filter
        self.B_t = nn.Linear(embedding_size, state_space_size, bias=False, dtype=T.complex64)
        
        # output filter
        self.C_t = nn.Linear(embedding_size, state_space_size, bias=False, dtype=T.complex64)

        ### output processing
        # normalizes the exponential channel from imaginary portion of the outputs
        self.norm = nn.LayerNorm(embedding_size)

        # embeds the output magnitude
        self.mag_embedding = nn.Linear(embedding_size, state_space_size, bias=False, dtype=T.float32)

        # embeds the imaginary output sin, cos, and theta derived from the phase angle
        self.phase_embedding = nn.Sequential(
            nn.Linear(embedding_size*3, state_space_size, bias=False, dtype=T.float32),
            nn.Tanh()
        )

        # compresses the output from state space size back into embedding size
        self.output_compression = nn.Sequential(
            nn.Linear(state_space_size, state_space_size//2, dtype=T.float32),
            nn.LayerNorm(state_space_size//2),
            nn.SiLU(),
            nn.Linear(state_space_size//2, embedding_size, dtype=T.float32),
            nn.SiLU()
        )

        # next state prediction
        self.state_prediction = nn.Sequential(
            nn.Linear(embedding_size, embedding_size//2, dtype=T.float32),
            nn.LayerNorm(embedding_size//2),
            nn.SiLU(),
            nn.Linear(embedding_size//2, output_size, dtype=T.float32),
        )

    # calculates the recurrent hidden state
    def forward(self, x_gated_t, delta_t, B_t, h_prev):
        
        A_prime_t = T.exp(-delta_t.unsqueeze(2) * self.state_transition_matrix.unsqueeze(0))
        B_prime_t = delta_t.unsqueeze(2) * B_t.unsqueeze(1) 
        h_prime_t = A_prime_t * h_prev + B_prime_t * x_gated_t.unsqueeze(-1)

        return h_prime_t

    def post_process_output(self, y_t):
        # features derived from complex output y_t
        theta_t = T.angle(y_t)
        exp_theta_t = self.norm(T.exp(theta_t))
        cos_t = T.cos(theta_t)
        sin_t = T.sin(theta_t)
        mag_t = T.abs(y_t)

        # combines phase features
        phase_t = T.concat(tensors=(exp_theta_t, cos_t, sin_t), dim=-1)
        # converts the complex y_t output into real values
        mag_proj = self.mag_embedding(mag_t)
        phase_proj = self.phase_embedding(phase_t)
        combined = mag_proj * phase_proj
        
        y_prime_t = self.output_compression(combined)

        return y_prime_t
    
    def pre_process_inputs(self, x_t):
        x_complex_t = T.complex(x_t, T.zeros_like(x_t))
        # embed inputs
        x_embed_t = self.embedding_layer(x_complex_t)
        
        # calculate input gates
        input_weights = self.input_gate(x_embed_t)
       
        # gate inputs
        x_gated_t = x_embed_t * input_weights

        # gate outputs
        output_weights = self.output_gate(x_gated_t)
        
        # modulate discretization matrix
        delta_t = self.discretization_matrix(x_gated_t)

        # modulate input/outputs
        B_t = self.B_t(x_gated_t)
        C_t = self.C_t(x_gated_t)

        return x_gated_t, delta_t, B_t, C_t, output_weights
    
    def construct_complex_matrix(self, real_values, img_values): 
        complex_matrix = T.exp(-T.exp(real_values))*T.exp(img_values * 1j)

        return complex_matrix
    
    def initialize_weights(self, low, high, dims, fn=None):
        weights = T.tensor(np.random.uniform(low=low, high=high, size=dims), dtype=T.float32, device=self.device)
        if fn is not None:
            weights = fn(weights)
        return weights
    
class RecurrentNetwork(nn.Module):
    def __init__(
            self,
            embedding_size,
            state_space_size,
            input_size,
            output_size,
            batch_size,
            device,
            chkpt_dir='temp/ppo',
            network_name='ssm'
            ):
        super(RecurrentNetwork, self).__init__()

        self.device = device
        self.state_space_size = state_space_size

        '''
        embedding size: the size of the expanded input feature space
        state_space_size: internal memory of the hidden state
        blocks: the number of recurrent blocks for multi-modal inputs. By default 1 for a single mode (WIP)
        sequential_mode: False if network processes entire sequence at once. True if network processes inputs sequentially. Affects training time
        '''
        # create save dir
        self.checkpoint_file = os.path.join(chkpt_dir, network_name)

        self.recurrent_block = RecurrentBlock(embedding_size=embedding_size, 
            state_space_size=state_space_size, 
            input_size=input_size, 
            batch_size=batch_size, 
            output_size=output_size, 
            device=self.device
        )

        # transfer all parameters into gpu if possible
        self.to(device=self.device)

    def recurrent_step(self, x_t, h_t, embeddings_only):

        # batch size of the current input. Used for variable batch sizes during inference and training
        
        
        # permute inputs to be of shape (L, B, D)
        x_permuted_t = x_t.permute(1,0,2) 
        inputs_t = self.recurrent_block.pre_process_inputs(x_permuted_t)

        # preallocate memory for outputs
        outputs = T.empty_like(input=inputs_t[0].permute(1,0,2), dtype=T.complex64, device=self.device) # (B, L, D)

        for idx, input_t in enumerate(zip(*inputs_t)):
            x_gated_t, delta_t, B_t, C_t, output_weights = input_t

            # calculate hidden next hidden state
            h_t = self.recurrent_block(x_gated_t, delta_t, B_t, h_t)

            # calculate output
            y_t = output_weights * (C_t.unsqueeze(1) * h_t).sum(dim=-1) + (1 - output_weights) * x_gated_t
            
            # save output
            outputs[:, idx, :] = y_t

        
        outputs = self.recurrent_block.post_process_output(outputs)

        if embeddings_only is True:
            return h_t, outputs
        else:
            return h_t, self.recurrent_block.state_prediction(outputs)
    
    def forward(self, x_t, h_t=None, embeddings_only=True):
        '''
        embeddings_only: controls if raw output embeddings or usable outputs should be returned
        '''
        if h_t is None:
            batch_size_t = x_t.shape[0]
            h_in = self.recurrent_block.hidden_state_matrix.unsqueeze(0).expand(batch_size_t, -1, -1)
        else:
            h_in = h_t

        return self.recurrent_step(x_t, h_in, embeddings_only)        
        
    def update_params(self):
        '''
        used to update initial hidden state and state transition matrix on learning step
        '''
        self.recurrent_block.state_transition_matrix = self.recurrent_block.construct_complex_matrix(
            real_values=self.recurrent_block.real_transition_matrix,
            img_values=self.recurrent_block.img_transition_matrix
        )

        self.recurrent_block.hidden_state_matrix = self.recurrent_block.construct_complex_matrix(
            real_values=self.recurrent_block.real_hidden_matrix,
            img_values=self.recurrent_block.img_hidden_matrix
        )

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
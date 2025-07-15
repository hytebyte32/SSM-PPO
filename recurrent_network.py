import os
import numpy as np
import torch as T
import torch.nn as nn

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
            duplicate_batchwise=True
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
    def forward(self, x_t, h_prev):
        
        x_gated_t, delta_t, B_t, C_t, output_weights = self.pre_process_inputs(x_t)

        A_prime_t = T.exp(-T.einsum("dn,bd->bdn", self.state_transition_matrix, delta_t))
        B_prime_t = T.einsum("bn,bd->bdn", B_t, delta_t)
        h_prime_t = T.einsum("bdn,bdn->bdn", A_prime_t, h_prev) + T.einsum("bdn,bd->bdn", B_prime_t, x_gated_t)
        
        y_t = output_weights * T.einsum("bn,bdn->bd", C_t, h_prime_t) + (1 - output_weights) * x_gated_t
        y_t = self.post_process_output(y_t)

        return h_prime_t, y_t

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
        #y_prime_t = self.output_compression(combined)

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
    
    def construct_complex_matrix(self, real_values, img_values, duplicate_batchwise=False): 
        complex_matrix = T.exp(-T.exp(real_values))*T.exp(img_values * 1j)

        if duplicate_batchwise:
            complex_matrix = complex_matrix.unsqueeze(0).expand(self.batch_size, -1, -1)

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

    def recurrent_step(self, x_t, h_t=None):
        # only being passed through linear layers so batch safe
        #*inputs_t, C_t, output_weights = self.recurrent_block.pre_process_inputs(x_t) # shaoe: (B, L, E or S)
        #x_gated_t = inputs_t[0]
        #input_permuted_t = [x.permute(1,0,2) for x in inputs_t] # converts to (L, B, E/S)
        x_permuted_t = x_t.permute(1,0,2)

        if h_t is None:
            h_prev = self.recurrent_block.hidden_state_matrix
        else:
            h_prev = h_t

        hidden_states = []
        outputs = []

        for x in x_permuted_t:
            #h_prev = checkpoint(self.recurrent_block.forward, use_reentrant=True)(*input_t, h_prev)
            h_prev, y_t = self.recurrent_block(x, h_prev)
            hidden_states.append(h_prev)
            outputs.append(y_t)

        # construct outputs
        hidden_states = T.stack(tensors=hidden_states).permute(1,0,2,3).to(device=self.device)
        outputs = T.stack(tensors=outputs).permute(1,0,2).to(device=self.device)      
        

        return hidden_states, outputs
    
    def forward(self, x_t, h_t, embeddings_only=True):
        h_out, y_out = checkpoint(self.recurrent_step, x_t, h_t, use_reentrant=True)
        if embeddings_only is True:
            return h_out, y_out
        else:
            return h_out, self.recurrent_block.state_prediction(y_out)    

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
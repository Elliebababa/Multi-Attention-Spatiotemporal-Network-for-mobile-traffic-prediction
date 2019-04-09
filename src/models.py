import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LSTM,RepeatVector,Dense,Activation,Add,Reshape,Input,Lambda,Multiply,Concatenate,Dot,Permute, Softmax,SimpleRNN
import keras.backend as K
latent_dim = 64

def RNN(lookback = 6, predstep = 3, input_dim = 1, latent_dim = latent_dim):
    #RNN
    #input shape [batch_size, timestep, input_dim]
    #output shape [batch_size, prestep, output_dim] #output_dim is 1 by default
    rnn_inputs = Input(shape=(None, input_dim),name = 'rnn_input') 
    rnn = SimpleRNN(latent_dim, name = 'RNN')
    rnn_r1 = rnn(rnn_inputs)
    rnn_outputs = RepeatVector(1)(rnn_r1)
    rnn_outputs = Dense(1)(rnn_outputs)
    rnn_model = Model(rnn_inputs, rnn_outputs)
    return rnn_model

def lstm(lookback = 6, predstep = 3, input_dim = 1, latent_dim = latent_dim):
    #LSTM
    #input shape [batch_size, timestep, input_dim]
    #output shape [batch_size, prestep, output_dim] #output_dim is 1 by default
    lstm_inputs = Input(shape=(None, input_dim),name = 'lstm_input') 
    lstm = LSTM(latent_dim, name = 'lstm')
    lstm_r1 = lstm(lstm_inputs)
    lstm_outputs = RepeatVector(1)(lstm_r1)
    lstm_outputs = Dense(1)(lstm_outputs)
    lstm_model = Model(lstm_inputs, lstm_outputs)
    return lstm_model

def seq2seq(lookback = 6, input_dim = 1, latent_dim = latent_dim):
    encoder_inputs = Input(shape = (lookback, input_dim), name = 'encoder_input')
    encoder = LSTM(latent_dim, return_state = True, name = 'encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape = (None, 1), name = 'decoder_input')
    decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True, name = 'decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)

    decoder_dense = Dense(1, name='output_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    #model for training
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    #encoder 
    #intput: endoer_inputs
    #output: encoder_states
    encoder_model = Model(encoder_inputs, encoder_states)
    
    #decoder for decode sequences
    #input: decoder_input, decoder_states_inputs
    #output: decoder_outputs, states
    decoder_state_input_h = Input(shape = (latent_dim,), name = 'decoder_ini_state_h')
    decoder_state_input_c = Input(shape = (latent_dim,), name = 'decoder_ini_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model

def lstm_prediction(test_input, model, pre_step = 1):
    #test_input (samples, steps, n)
    #lstm seq
    lstm_i_seq = test_input[0]
    lstm_o_seq = np.zeros((test_input[0].shape[0], pre_step, 1))
    for i in range(pre_step):
        output = model.predict(lstm_i_seq)
        lstm_o_seq[:,i,0] = output[:,0,0]
        #update the target sequence of length 1
        lstm_i_seq[:,0:5,0] = lstm_i_seq[:,1:6,0]
        lstm_i_seq[:,5,0] = output[:,0,0]
    return lstm_o_seq
    
class MASTNN(object):
    def __init__(self, T = 6, predT = 1, encoder_latent_dim = 64, decoder_latent_dim = 64, aux_att = True, global_att = True, neigh_num = 15, semantic = False, trainmode = False):
        super(MASTNN, self).__init__()
        self.trainmode = trainmode # when trainmode is no, the decoder will use the truth value of prev time step to decode
        self.T = T
        self.predT = predT
        self.encoder_latent_dim = encoder_latent_dim
        self.decoder_latent_dim = decoder_latent_dim
        #lstm for encoder and decoder 
        self.enLSTM = LSTM(encoder_latent_dim, return_state = True, name = 'encoder_lstm')
        self.deLSTM = LSTM(decoder_latent_dim, return_state = True, name = 'decoder_lstm')
        #encoder local attention parameter weight matrix
        self.We = Dense(units = T, input_dim = 2 * encoder_latent_dim, activation = 'linear', use_bias = False, name = 'We')
        self.Ue = Dense(units = T, input_dim = T, activation = 'linear', use_bias = False, name = 'Ue')
        self.Ve = Dense(units = 1, input_dim = T, activation = 'linear', use_bias = False, name = 'Ve')
        #encooder global attention parameter weight matrix
        self.global_att = global_att
        self.neigh_num = neigh_num
        self.lamb = 0.4
        print('lambda:',self.lamb)
        self.Wg = Dense(units = T, input_dim = 2 * encoder_latent_dim, activation = 'linear', use_bias = False, name = 'Wg')
        self.Ug = Dense(units = T, input_dim = T, activation = 'linear', use_bias = False, name = 'Ug')
        self.ug = Dense(units = 1, input_dim = T, activation = 'linear', use_bias = False, name = 'ug')
        self.Wg_ = Dense(units = T, input_dim = neigh_num, activation = 'linear', use_bias = False, name = 'Wg_')
        self.Vg = Dense(units = 1, input_dim = T, activation = 'linear', use_bias = False, name = 'Vg')
        #decoder temporal attention parameter weight matrix
        self.semantic = semantic
        self.Wd = Dense(units = decoder_latent_dim, input_dim = decoder_latent_dim, activation = 'linear', use_bias = False, name = 'Wd')
        self.Wd_ = Dense(units = decoder_latent_dim, input_dim = 2*encoder_latent_dim, activation = 'linear', use_bias = False, name = 'Wd_')
        self.Vd = Dense(units = 1, input_dim = decoder_latent_dim, activation = 'linear', use_bias = False, name = 'Vd')
        #output parameter matrix
        #self.Wo1 = Dense(units = 64, activation = 'sigmoid', use_bias = True, name = 'Dense1_for_output')
        self.Wo2 = Dense(units = 1, activation = 'sigmoid', use_bias = True, name = 'Dense2_for_output')
        
    def spatial_attention(self,encoder_inputs, enc_attn, init_states):
        # input : encoder_inputs [batch_size, time_steps, input_dim], init_states,
        #            att_flag: 1:only local, 2:local+ global
        # return : encoder_output, encoder_state, encoder_att
        [h,s] = init_states
        encoder_att = []
        encoder_output = []
        global_att = self.global_att
        #get input
        if global_att:
            local_inputs = encoder_inputs[0]
            global_inputs = encoder_inputs[1]
        else:
            local_inputs = encoder_inputs
        
        #get att states
        if global_att:
            local_attn = enc_attn[0]
            global_attn = enc_attn[1]
        else:
            local_attn = enc_attn
        
        #local attention
        #shared layer
        AddLayer = Add(name = 'add')
        PermuteLayer = Permute(dims = (2,1))
        ActTanh = Activation(activation = 'tanh',name ='tanh_for_e')
        ActSoftmax = Activation(activation = 'softmax', name ='softmax_for_alpha')
        def local_attention(states,step):
            #for attention query
            #linear map
            Ux = self.Ue(PermuteLayer(local_inputs)) #[none,input_dim,T]
            states = Concatenate(axis = 1, name = 'state_{}'.format(step))(states)
            Whs = self.We(states) #[none, T]
            Whs = RepeatVector(local_inputs.shape[2])(Whs) #[none,input_dim,T]
            y = AddLayer([Ux, Whs])
            e = self.Ve(ActTanh(y)) #[none,input_dim,1]
            e = PermuteLayer(e) #[none,1,input_dim]
            alpha = ActSoftmax(e)
            return alpha
        
        AddLayer2 = Add(name = 'add2')
        PermuteLayer2 = Permute(dims = (2,1))
        ActTanh2 = Activation(activation = 'tanh',name ='tanh_for_e2')
        ActSoftmax2 = Activation(activation = 'softmax', name ='softmax_for_beta')
        
        def global_attention_v2(states, step, prior):
            #for global attention query
            #global inputs[0](values) [none, T, neighbornum]
            #linear map Wg
            states = Concatenate(axis = 1, name = 'state_gl_{}'.format(step))(states)
            Wgs = self.Wg(states) #[none,T]
            Ugy = self.Ug(PermuteLayer2(global_inputs[0])) # [none, neighbornum, T]
            #Wxu = self.ug(PermuteLayer2(global_inputs[0])) # [none, neighbornum, 1]
            #Wgx = self.Wg_(Wxu) # [none,neighbornum,T]
            Wgs_ = RepeatVector(global_inputs[0].shape[2])(Wgs) # [none, neighbornum, T]
            y2 = AddLayer2([Wgs_,Ugy])
            g = self.Vg(ActTanh2(y2))
            g = PermuteLayer2(g)
            g_ = Lambda(lambda x: (1-self.lamb)*x + self.lamb*prior)(g)
            beta = ActSoftmax2(g)
            return beta
        
        for t in range(self.T):
            if not global_att:
                x = Lambda(lambda x: x[:,t ,:], name = 'X_{}'.format(t))(local_inputs) #[none,input_dim]
                x = RepeatVector(1)(x) #[none,1,input_dim] , 1 denotes one time step
                local_x = Multiply(name = 'Xatt_{}'.format(t))([local_attn, x]) #[none,1,input_dim]
                o, h, s = self.enLSTM(local_x, initial_state = [h, s]) #o, h, s [none, hidden_dim]
                o = RepeatVector(1)(o)
                encoder_output.append(o)
                local_attn = local_attention([h,s], t+1)
                encoder_att.append(local_attn)
            elif global_att:
                x = Lambda(lambda x: x[:,t ,:], name = 'X_local_{}'.format(t))(local_inputs) #[none,input_dim]
                x = RepeatVector(1)(x) #[none,1,input_dim] , 1 denotes one time step
                [global_input_value,global_input_weight] = global_inputs
                x2 = Lambda(lambda x2: x2[:,t ,:], name = 'X_global_{}'.format(t))(global_input_value) #[none,neighbornum]
                x2 = RepeatVector(1)(x2) #[none,1,neighbornum] , 1 denotes one time step
                prior = Lambda(lambda p: p[:,t ,:], name = 'global_prior_{}'.format(t))(global_input_weight) #[none,neighbornum]
                prior = RepeatVector(1)(prior)
                local_x = Multiply(name = 'Xatt_local_{}'.format(t))([local_attn, x]) #[none,1,input_dim]
                print('global_attn:',global_attn, 'x2:',x2)
                global_x = Dot(axes = (2),name = 'Xatt_global_{}'.format(t))([global_attn, x2])
                #global_x = Multiply(name = 'Xatt_global_{}'.format(t))([global_attn, x2])
                att_x = Concatenate(axis = -1)([local_x, global_x])
                o, h, s = self.enLSTM(att_x, initial_state = [h, s]) #o, h, s [none, hidden_dim]
                o = RepeatVector(1)(o)
                encoder_output.append(o)
                local_attn = local_attention([h,s], t+1)
                global_attn = global_attention_v2([h,s],t+1, prior)
                encoder_att.append([local_attn, global_attn])
        
        if not global_att: 
            encoder_att = Concatenate(axis = 1,name = 'encoder_att')(encoder_att) #[none, T, input_dim]
        
        else:
            local_att = [i[0] for i in encoder_att]
            print('local_att', local_att)
            local_att = Concatenate(axis = 1)(local_att)
            global_att = [i[1] for i in encoder_att]
            print('global_att', global_att)
            #global_att = Concatenate(axis = 1)(global_att)
            global_att = Lambda(lambda x: K.concatenate(x, axis = 1))(global_att)
            encoder_att = [local_att, global_att]
        
        encoder_output = Concatenate(axis = 1, name = 'encoder_output')(encoder_output)
        
        return encoder_output, [h,s], encoder_att
    
    def temporal_attention(self, decoder_inputs,initial_state,attention_states):
        #input : decoder_inputs, intial_state, attention_states
        #return : output, state
        
        #get input
        if self.semantic:
            last_inputs = decoder_inputs[0]
            semantic_inputs = decoder_inputs[1]
        else:
            last_inputs = decoder_inputs
        
        
        AddLayer = Add(name = 'add_tem')
        PermuteLayer = Permute(dims = (2,1))
        ActTanh = Activation(activation = 'tanh',name ='tanh_for_d')
        def attention(states, step):
            #return context for the t step
            Wh = self.Wd(attention_states) #[none, T, latent_dim]
            states = Concatenate(axis = 1, name = 'state_hat_{}'.format(step))(states)
            Wds = self.Wd_(states) #[none, latent_dim]
            Wds = RepeatVector(attention_states.shape[1])(Wds) #[none, T, latent_dim]
            y = AddLayer([Wds,Wh])
            u = self.Vd(ActTanh(y))#[none, T, 1]
            a = Softmax(axis = 1)(u)
            c = Dot(axes = (1))([a,attention_states]) #[none, 1, latent_dim], the summed context over encoder outputs at certain pred time step 
            return c
        
        
        [h,s] = initial_state
        context = attention([h,s],0) 
        outputs =[]
        prev = None
        for t in range(self.predT):
            if not self.trainmode and t > 0 and prev is not None:
            #if decoder length is larger than 1, we calculate the prediction output for current step
                last_pred = self.Wo2(prev)
            else:
                last_pred = Lambda(lambda x: x[:,t ,:], name = 'X_tem{}'.format(t))(last_inputs) #[none,decoder_input_dim]
                last_pred = RepeatVector(1)(last_pred) #[none,1,decoder_input_dim] , 1 denotes one time step
            
            #if self.semantic:
            #    x = Concatenate(axis = -1)([context,last_pred, semantic_inputs])
            #else:
            x = Concatenate(axis = -1)([context,last_pred])
            o, h, s = self.deLSTM(x, initial_state = [h, s])
            context = attention([h,s],t+1)#[none, 1, latent_dim]
            o = RepeatVector(1)(o)
            if self.semantic:
            #    print('semantic input: ',semantic_inputs)
            #    print('o: ',o)
                o = Concatenate(axis = -1)([o, semantic_inputs])
            outputs.append(o)
            prev = o
            
        if len(outputs) > 1:
            outputs = Concatenate(axis = 1, name = 'decoder_output')(outputs)
        else:
            outputs = tf.convert_to_tensor(outputs[0])
            
        print(outputs)
        return outputs,[h,s]
        
        
    def build_model(self, input_dim = int(5)):
        #encoder
        encoder_latent_dim = self.encoder_latent_dim
        neighnum = self.neigh_num
        T = self.T
        h0 = Input(shape = (encoder_latent_dim,),name = 'h_initial')
        s0 = Input(shape = (encoder_latent_dim,),name = 's_initial')
        enc_att_local = Input(shape = (1,input_dim),name = 'enc_att_local')
        enc_att_global = Input(shape = (1,neighnum),name = 'enc_att_global')
        encoder_inputs_local = Input(shape = (T,input_dim), name = 'encoder_input_local')
        encoder_inputs_global_value = Input(shape = (T, neighnum), name = 'encoder_input_global_value')
        encoder_inputs_global_weight = Input(shape = (T, neighnum), name = 'encoder_input_global_weight')
        if self.global_att:
            encoder_inputs = [encoder_inputs_local,[encoder_inputs_global_value,encoder_inputs_global_weight]]
            enc_att = [enc_att_local,enc_att_global]
        else:
            encoder_inputs = encoder_inputs_local
            enc_att = enc_att_local
        encoder_output, encoder_state, encoder_att = self.spatial_attention(encoder_inputs,enc_att,[h0, s0])
               
        #decoder
        dim = 1
        last_inputs = Input(shape = (None, dim))
        semantic_inputs = Input(shape = (1, 10))
        if self.semantic:
            decoder_inputs = [last_inputs,semantic_inputs]
        else:
            decoder_inputs = last_inputs
        decoder_outputs, states = self.temporal_attention(decoder_inputs, encoder_state, encoder_output)
        
        #linear transform
        #output = Dense(1, activation = 'linear', name = 'output_dense')(decoder_att)
        #output = self.Wo1(decoder_outputs)
        output = self.Wo2(decoder_outputs)
        if not self.global_att:
            if not self.semantic:
                model = Model([encoder_inputs, h0, s0, enc_att, last_inputs], output)
            else:
                model = Model([encoder_inputs, h0, s0, enc_att, last_inputs, semantic_inputs], output)
        else:
            if not self.semantic:
                model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, last_inputs], output)
            else:
                model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, last_inputs, semantic_inputs], output)
            
        return model

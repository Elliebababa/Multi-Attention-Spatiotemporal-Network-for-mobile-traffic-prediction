import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LSTM,RepeatVector,Dense,Activation,Add,Reshape,Input,Lambda,Multiply,Concatenate,Dot,Permute, Softmax
import keras.backend as K
latent_dim = 64
dropout = 0
lookback = 6

def lstm(lookback = lookback, predstep = 3, latent_dim = latent_dim, dropout = dropout):
    #LSTM
    lstm_inputs = Input(shape=(None, 1),name = 'lstm_input') 
    lstm = LSTM(latent_dim, dropout= dropout, name = 'lstm')
    lstm_r1 = lstm(lstm_inputs)
    lstm_outputs = RepeatVector(predstep)(lstm_r1)
    lstm_outputs = Dense(1)(lstm_outputs)
    lstm_model = Model(lstm_inputs, lstm_outputs)
    return lstm_model

def seq2seq(lookback = lookback, latent_dim = latent_dim, dropout = dropout):
    encoder_inputs = Input(shape = (lookback, 1), name = 'encoder_input')
    encoder = LSTM(latent_dim, return_state = True, name = 'encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape = (None, 1), name = 'decoder_input')
    decoder_lstm = LSTM(latent_dim, dropout = dropout,  return_sequences = True, return_state = True, name = 'decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)

    decoder_dense = Dense(1, name='output_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    #model for training
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    #encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    #decoder
    decoder_state_input_h = Input(shape = (latent_dim,), name = 'decoder_ini_state_h')
    decoder_state_input_c = Input(shape = (latent_dim,), name = 'decoder_ini_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model


def seq2seq_aux(lookback = lookback, latent_dim = latent_dim, dropout = dropout):
    encoder_inputs = Input(shape = (lookback, 1), name = 'encoder_input')
    encoder_inputs_aux = Input(shape=(lookback, aux_dimension),name = 'encoder_input_aux') 
    encoder_inputs2 = Concatenate(axis = -1,name = 'encoder_input_concated_with_aux')([encoder_inputs,encoder_inputs_aux])
    encoder2 = LSTM(latent_dim, dropout= dropout, return_state=True, name='encoder_lstm2')
    encoder_outputs2, state_h2, state_c2 = encoder2(encoder_inputs2)

    encoder_states2 = [state_h2, state_c2]

    decoder_inputs = Input(shape=(None, 1)) 

    decoder_lstm2 = LSTM(latent_dim,return_sequences=True, return_state=True, name='decoder_lstm2')
    decoder_outputs2, _, _ = decoder_lstm2(decoder_inputs, initial_state=encoder_states2)

    decoder_dense2 = Dense(1,name='decoder_dense2') # 1 continuous output at each timestep
    decoder_outputs2 = decoder_dense2(decoder_outputs2)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model2 = Model([encoder_inputs, encoder_inputs_aux,decoder_inputs], decoder_outputs2)

    encoder_model2 = Model([encoder_inputs,encoder_inputs_aux], encoder_states2)

    #the decoder stage takes in predicted targer inputs and encoded state vectors, return predicted target outputs and decode state vecotrs
    decoder_state_input_h2 = Input(shape = (latent_dim,),name = 'decoder_ini_state_h2')
    decoder_state_input_c2 = Input(shape = (latent_dim,),name = 'decoder_ini_state_c2')
    decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]

    decoder_outputs2, state_h2, state_c2 = decoder_lstm2(decoder_inputs, initial_state = decoder_states_inputs2)
    decoder_states2 = [state_h2, state_c2]

    decoder_outputs2 = decoder_dense2(decoder_outputs2)
    decoder_model2 = Model([decoder_inputs] + decoder_states_inputs2, [decoder_outputs2]+decoder_states2)

    return model2, encoder_model2, decoder_model2


def seq2seq_aux_his(test_input, encoder_model, decoder_model, decoder_aux = None):
    aux_dimension = 4
    #define an input series and encode it with as LSTM
    #encoder_inputs_aux = Input(shape=(None, aux_dimension),name = 'encoder_input_aux') 
    #encoder_inputs2 = Concatenate(axis = -1,name = 'encoder_input_concated_with_aux')([encoder_inputs,encoder_inputs_aux])
    encoder3 = LSTM(latent_dim, dropout= dropout, return_state=True, name='encoder_lstm3')
    encoder_outputs3, state_h3, state_c3 = encoder3(encoder_inputs2)

    encoder_states3 = [state_h3, state_c3]

    decoder_inputs = Input(shape=(None, 1)) 
    decoder_historical_input = Input(shape=(None,2),name = 'decoder_historical_input')
    decoder_inputs3 = Concatenate(axis = -1,name = 'decoder_input_concated_with_his')([decoder_inputs,decoder_historical_input])
    decoder_lstm3 = LSTM(latent_dim,return_sequences=True, return_state=True, name='decoder_lstm3')
    decoder_outputs3, _, _ = decoder_lstm3(decoder_inputs3, initial_state=encoder_states3)

    decoder_dense3 = Dense(1,name='decoder_dense3') # 1 continuous output at each timestep
    decoder_outputs3 = decoder_dense3(decoder_outputs3)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model3 = Model([encoder_inputs, encoder_inputs_aux,decoder_inputs,decoder_historical_input], decoder_outputs3)

    encoder_model3 = Model([encoder_inputs,encoder_inputs_aux], encoder_states3)

    #the decoder stage takes in predicted targer inputs and encoded state vectors, return predicted target outputs and decode state vecotrs
    decoder_state_input_h3 = Input(shape = (latent_dim,),name = 'decoder_ini_state_h3')
    decoder_state_input_c3 = Input(shape = (latent_dim,),name = 'decoder_ini_state_c3')
    decoder_states_inputs3 = [decoder_state_input_h3, decoder_state_input_c3]

    decoder_outputs3, state_h3, state_c3 = decoder_lstm3(decoder_inputs3, initial_state = decoder_states_inputs3)
    decoder_states3 = [state_h3, state_c3]

    decoder_outputs3 = decoder_dense3(decoder_outputs3)
    decoder_model3 = Model([decoder_inputs,decoder_historical_input] + decoder_states_inputs3, [decoder_outputs3]+decoder_states3)
    return model3, encoder_model3, decoder_model3


def seq2seq_att(test_input, encoder_model, decoder_model, decoder_aux = None):
    T = lookback
    series_dim = 5
    We = Dense(units = T , input_dim = 2*latent_dim, activation=None, use_bias=False,name = 'We') # without activation and bias, pure shared kernel, Weight e (T*2m)
    Ue = Dense(units = T , input_dim = T,activation=None, use_bias=False,name = 'Ue') 
    Ve = Dense(units = 1, input_dim = T, activation=None, use_bias=False,name = 'Ve') 
    enLstm = LSTM(latent_dim,return_state=True,name = 'encoder_lstm4')

    def compute_attention(h_prev,s_prev,X,step):
        '''
        :compute attention alpha for each time step
        :param h_prev: previous hidden state (None,latent_dim, )
        :param s_prev: previous cell state (None,latent_dim, )
        :param X: (None, T, n),n is length of input series at time t,T is length of time series
        :return: x_t's attention weights,total n numbers,sum these are 1
        '''

        prev = Concatenate(axis = 1, name = 'pre_state_{}'.format(step))([h_prev,s_prev])  #(None,2*latent_dim,)
        r1 = We(prev)   #(none,T,)
        r1 = RepeatVector(X.shape[-1],name = 'repeat_x_at_{}'.format(step))(r1)  #(none,n,T)
        X_temp =  Permute(dims=(2,1),name = 'X_tmp_at_{}'.format(step))(X) #X_temp(None,n,T)
        r2 = Ue(X_temp)  # (none,n,T)  Ue(T,T)
        r3 = Add()([r1,r2])  #(none,n,T)
        r4 = Activation(activation='tanh',name = 'act_at_{}'.format(step))(r3)  #(none,n,T)
        r5 = Ve(r4) #(none,n,1)
        r5 = Permute(dims=(2,1),name = 'e_at_{}'.format(step))(r5) #(none,1,n)
        alphas = Activation(activation='softmax',name = 'get_attention_{}'.format(step))(r5)
        return alphas

    def attentionX(X,h0,s0):
        '''
        convert the origin input to attentioned one , the step includes 
        : 1 running the lstm to encode and get the hidden layer; 2 compute attention;3 update x
        : X (none, T, n)
        : s0 (latent_dim,)
        : h0 (latent_dim,)
        '''
        T = lookback#X.shape[1]
        h =  h0 #np.zerso(shape = (latent_dim,))
        s =  s0 #np.zerso(shape = (latent_dim,))
        #initialize empty list of outputs
        attention_weight_t = None
        for t in range(T):
            alphas = compute_attention(h, s, X,t)  #(none,1,n)
            x = Lambda(lambda x: x[:,t,:], name = 'X_{}'.format(t))(X) 
            x = Reshape((1,series_dim))(x) #(none,1,n)
            h, _, s = enLstm(x, initial_state=[h, s]) 
            if t != 0:
                print(t,attention_weight_t)
                print(alphas)
                attention_weight_t = Lambda(lambda x:K.concatenate(x[:],axis=1))([attention_weight_t,alphas])
                #attention_weight_t = Concatenate(axis = 1)([attention_weight_t,alphas])
            else:
                attention_weight_t = alphas
        #got attention_weight (none, T, n)
        X_ = Multiply(name = 'attention_X')([attention_weight_t,X])
        return X_

        #define an input series and encode it with as LSTM
        h0 = Input(shape = (latent_dim,),name='h_initial')
        s0 = Input(shape = (latent_dim,),name='s_initial')
        encoder_inputs = Input(shape=(T, 1),name = 'encoder_input') 
        encoder_inputs_aux = Input(shape=(T, aux_dimension),name = 'encoder_input_aux') 
        encoder_inputs2 = Concatenate(axis = -1,name = 'encoder_input_concated_with_aux')([encoder_inputs,encoder_inputs_aux]) 
        encoder_att_input = attentionX(encoder_inputs2,h0,s0)
        encoder5 = enLstm #LSTM(latent_dim,return_state=True,name = 'encoder_lstm4')
        encoder_outputs5, state_h5, state_c5 = encoder5(encoder_att_input)

        encoder_states5 = [state_h5, state_c5]
        decoder_inputs = Input(shape=(None, 1))
        decoder_historical_input = Input(shape=(None,2),name = 'decoder_historical_input')
        decoder_inputs5 = Concatenate(axis = -1,name = 'decoder_input_concated_with_his')([decoder_inputs,decoder_historical_input])
        decoder_lstm5 = LSTM(latent_dim,return_sequences=True, return_state=True, name='decoder_lstm5')
        decoder_outputs5, _, _ = decoder_lstm5(decoder_inputs5, initial_state=encoder_states5)

        decoder_dense5 = Dense(1,name='decoder_dense5') # 1 continuous output at each timestep
        decoder_outputs5 = decoder_dense5(decoder_outputs5)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model5 = Model([encoder_inputs, encoder_inputs_aux, h0, s0,decoder_inputs,decoder_historical_input], decoder_outputs5)

        #from the previous model - mapping encoder sequence to state vecotrs
        encoder_model5 = Model([encoder_inputs,encoder_inputs_aux,h0,s0], encoder_states5)

        #the decoder stage takes in predicted targer inputs and encoded state vectors, return predicted target outputs and decode state vecotrs
        decoder_state_input_h5 = Input(shape = (latent_dim,),name = 'decoder_ini_state_h5')
        decoder_state_input_c5 = Input(shape = (latent_dim,),name = 'decoder_ini_state_c5')
        decoder_states_inputs5 = [decoder_state_input_h5, decoder_state_input_c5]

        decoder_outputs5, state_h5, state_c5 = decoder_lstm5(decoder_inputs5, initial_state = decoder_states_inputs5)
        decoder_states5 = [state_h5, state_c5]

        decoder_outputs5 = decoder_dense5(decoder_outputs5)
        decoder_model5 = Model([decoder_inputs,decoder_historical_input] + decoder_states_inputs5, [decoder_outputs5]+decoder_states5)
        return model5, encoder_model5, decoder_model5


def decoder_prediction(test_input, encoder_model, decoder_model, pre_step = 1, decoder_aux = None):
    #test_input (samples，T, n)
    states_values = encoder_model.predict(test_input)
    target_seq = np.zeros((test_input[0].shape[0],1,1))
    target_seq[:,0,0] =  test_input[0][:,-1,0]

    decoded_seq = np.zeros((test_input[0].shape[0], pre_step, 1))

    for i in range(pre_step):
        if not decoder_aux is None:
            tmp = np.expand_dims(decoder_aux[:,i,], axis = 1)
            target_seq = [target_seq,tmp]
        else:
            target_seq = [target_seq]
        output, h, c = decoder_model.predict(target_seq + states_values)
        decoded_seq[:,i,0] = output[:,0,0]
        #update the target sequence of length 1
        target_seq = np.zeros((test_input[0].shape[0],1,1))
        target_seq[:,0,0] = output[:,0,0]
        #update states
        states_values = [h,c]
    return decoded_seq
    
class MAModel(object):
    def __init__(self, T = 6, encoder_latent_dim = 64, decoder_latent_dim = 64, global_att = False, neigh_num = 15):
        super(MAModel, self).__init__()
        self.T = T
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
        self.lamb = 0.3
        self.Wg = Dense(units = T, input_dim = 2 * encoder_latent_dim, activation = 'linear', use_bias = False, name = 'Wg')
        self.ug = Dense(units = 1, input_dim = T, activation = 'linear', use_bias = False, name = 'Ug')
        self.Wg_ = Dense(units = T, input_dim = neigh_num, activation = 'linear', use_bias = False, name = 'Wg_')
        self.Vg = Dense(units = 1, input_dim = T, activation = 'linear', use_bias = False, name = 'Vg')
        #decoder temporal attention parameter weight matrix
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
        def global_attention(states, step, prior):
            #for global attention query
            #global inputs [none, T, neighbornum]
            #linear map Wg
            states = Concatenate(axis = 1, name = 'state_gl_{}'.format(step))(states)
            Wgs = self.Wg(states) #[none,T]
            Wxu = self.ug(PermuteLayer2(global_inputs[0])) # [none, neighbornum, 1]
            Wgx = self.Wg_(Wxu) # [none,neighbornum,T]
            Wgx = RepeatVector(global_inputs[0].shape[2])(Wgs) # [none, neighbornum, T]
            y2 = AddLayer2([Wgs,Wgx])
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
                #print('global_attn:',global_attn, 'x2:',x2)
                global_x = Multiply(name = 'Xatt_global_{}'.format(t))([global_attn, x2])
                att_x = Concatenate(axis = -1)([local_x, x2])
                o, h, s = self.enLSTM(att_x, initial_state = [h, s]) #o, h, s [none, hidden_dim]
                o = RepeatVector(1)(o)
                encoder_output.append(o)
                local_attn = local_attention([h,s], t+1)
                global_attn = global_attention([h,s],t+1, prior)
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
        
        predT = 1
        [h,s] = initial_state
        context = attention([h,s],0) 
        outputs =[]
        for t in range(predT):
            x = Lambda(lambda x: x[:,t ,:], name = 'X_tem{}'.format(t))(decoder_inputs) #[none,decoder_input_dim]
            x = RepeatVector(1)(x) #[none,1,decoder_input_dim] , 1 denotes one time step
            x = Concatenate(axis = -1)([context,x])
            o, h, s = self.deLSTM(x, initial_state = [h, s])
            context = attention([h,s],t+1)#[none, 1, latent_dim]
            o = RepeatVector(1)(o)
            outputs.append(o)
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
        decoder_inputs = Input(shape = (None, dim))
        decoder_outputs, states = self.temporal_attention(decoder_inputs, encoder_state, encoder_output)
        
        #linear transform
        #output = Dense(1, activation = 'linear', name = 'output_dense')(decoder_att)
        #output = self.Wo1(decoder_outputs)
        output = self.Wo2(decoder_outputs)
        if not self.global_att:
            model = Model([encoder_inputs, h0, s0, enc_att, decoder_inputs], output)
        else:
            model = Model([encoder_inputs_local, encoder_inputs_global_value,encoder_inputs_global_weight, h0, s0, enc_att_local, enc_att_global, decoder_inputs], output)
        return model

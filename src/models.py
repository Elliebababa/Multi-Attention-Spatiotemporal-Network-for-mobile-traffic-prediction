import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LSTM,RepeatVector,Dense,Activation,Add,Reshape,Input,Lambda,Multiply,Concatenate,Dot,Permute
latent_dim = 64
dropout = 0
lookback = 6

def lstm(lookback = lookback, latent_dim = latent_dim, dropout = dropout):
    #LSTM
    lstm_inputs = Input(shape=(None, 1),name = 'lstm_input') 
    lstm = LSTM(latent_dim, dropout= dropout, name = 'lstm')
    lstm_r1 = lstm(lstm_inputs)
    lstm_outputs = Dense(1)(lstm_r1)
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


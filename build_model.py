# def build_model(data_splitting):
#     train_padded, test_padded, train_labels_encoded, test_labels_encoded, input_size, maxlen = data_splitting
#
#     # Define the model
#     model = Sequential()
#
#     # Add an embedding layer
#     model.add(Embedding(input_dim=input_size, output_dim=100, input_shape=(56,)))
#
#     # Add a bidirectional LSTM layer with 128 units
#     model.add(Bidirectional(LSTM(128, kernel_regularizer=l2(0.1), return_sequences=True, recurrent_regularizer=l2(0.1))))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#
#     # Add another LSTM layer
#     model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))))
#     # Add batch normalization layer
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#
#     # Add a dense layer with 64 units and ReLU activation
#     model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
#
#     # Add dropout regularization
#     model.add(Dropout(0.5))
#
#     # Add the output layer with 5 units for 5 labels and softmax activation
#     model.add(Dense(5, activation='softmax'))

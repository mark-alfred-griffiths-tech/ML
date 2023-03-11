class RecoverWinningVAE:
    def __init__(self, data, seed_value, batch_size, num_epochs, model_dir_encoder, model_dir_decoder,
                 project_name_encoder, project_name_decoder,
                 *args, **kwargs):
        super(RecoverWinningVAE, self).__init__(*args, **kwargs)
        self.data = data
        self.seed_value = seed_value
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_dir_encoder = model_dir_encoder
        self.model_dir_decoder = model_dir_decoder
        self.project_name_encoder = project_name_encoder
        self.project_name_decoder = project_name_decoder
        self.selector = None
        self.decoder = None
        self.best_hps_decoder = None
        self.best_hps_encoder = None
        self.overall_reconstruction()

    def overall_reconstruction(self):
        self.retrieve_best_hps_decoder_selector()
        self.selector = self.create_vae()
        return self

    def reconstruct_decoder(self):
        SetSeed(1234)
        self.decoder = Sequential()
        self.decoder.add(Dense(self.best_hps_decoder.get('ni_neurons_num'), activation=self.best_hps_decoder.get('ni_neurons_activation'),
                          kernel_initializer=glorot_uniform_initializer(),
                          bias_initializer='zeros',
                          input_shape=(self.data.dim_x,),
                          name='ni_neurons_layer'))
        self.decoder.add(Dense(self.best_hps_decoder.get('nj_neurons_num'), activation=self.best_hps_decoder.get('nj_neurons_activation'),
                          kernel_initializer=glorot_uniform_initializer(),
                          bias_initializer='zeros',
                          name='nj_neurons_layer'))
        self.decoder.add(Dense(self.best_hps_decoder.get('nk_neurons_num'), activation=self.best_hps_decoder.get('nk_neurons_activation'),
                          kernel_initializer=glorot_uniform_initializer(),
                          bias_initializer='zeros',
                          name='nk_neurons_layer'))
        self.decoder.add(Dense(self.best_hps_decoder.get('nl_neurons_num'), activation=self.best_hps_decoder.get('nl_neurons_activation'),
                          kernel_initializer=glorot_uniform_initializer(),
                          bias_initializer='zeros',
                          name='nl_neurons_layer'))
        self.decoder.add(Dense(self.best_hps_decoder.get('nm_neurons_num'), activation=self.best_hps_decoder.get('nm_neurons_activation'),
                          kernel_initializer=glorot_uniform_initializer(),
                          bias_initializer='zeros',
                          name='nm_neurons_layer'))
        self.decoder.add(Dense(self.data.num_cats, activation=tf.keras.activations.softmax,
                          kernel_initializer=glorot_uniform_initializer(),
                          bias_initializer='zeros',
                          name='softmax_layer'))
        self.decoder.compile(loss=tf.keras.losses.CategoricalCrossentropy(fproject_name_encoderrom_logits=False),
                        optimizer=SGD(momentum=self.best_hps_decoder('optimizer_momentum_float_value'),
                                      clipnorm=self.best_hps_decoder('optimizer_clipnorm_float_value')),
                        metrics=['accuracy'])
        return self

    def reconstruct_encoder(self):
        self.encoder = ConcreteAutoencoderFeatureSelector(K=self.best_hps_encoder.get('num_feats'), output_function=self.decoder,
                                                      num_epochs=self.num_epochs)
        return self

    def run_tuner_get_best_hyperparameters_encoder(self):
        tuner = kt.Hyperband(HyperEncoder(batch_size=self.batch_size), objective='accuracy', max_epochs=epochs, factor=3,
                             direbest_hps_decoderctory=self.model_dir_encoder,
                             project_name=self.project_name_encoder)

        self.best_hps_encoder = tuner.get_best_hyperparameters(1)[0]
        return self


    def run_tuner_get_best_hyperparameters_decoder(self):
        tuner = kt.Hyperband(HyperDecoder(batch_size=self.batch_size), objective='accuracy', max_epochs=epochs, factor=3,
                             direbest_hps_decoderctory=self.model_dir_decoder,
                             project_name=self.project_name_decoder)

        self.best_hps_decoder = tuner.get_best_hyperparameters(1)[0]
        return self

    def retrieve_best_hps_decoder_selector(self):
        self.best_hps_decoder = self.run_tuner_get_best_hyperparameters_decoder()
        self.best_hps_selector = self.run_tuner_get_best_hyperparameters_encoder()
        return self

    def create_vae(self):
        self.decoder = self.reconstruct_decoder()
        self.encoder = self.reconstruct_encoder()
        return self

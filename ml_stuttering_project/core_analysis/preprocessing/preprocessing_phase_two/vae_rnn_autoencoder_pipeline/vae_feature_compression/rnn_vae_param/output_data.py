import os


class output_data:
    def __init__(self, initialise_settings, output, x_full_z_y_df, *args, **kwargs):
        super(output_data, self).__init__(*args, **kwargs)
        self.dim_one_encoder = initialise_settings.dim_one_encoder
        self.dim_two_encoder = initialise_settings.dim_two_encoder
        self.dim_three_encoder = initialise_settings.dim_three_encoder
        self.dim_one_decoder = initialise_settings.dim_one_decoder
        self.dim_two_decoder = initialise_settings.dim_two_decoder
        self.dim_three_decoder = initialise_settings.dim_three_decoder
        self.dim_z = initialise_settings.dim_z
        self.output = output
        self.get_name()
        self.export_processed_file(x_full_z_y_df)

    def get_name(self):
        self.name = 'type_2_' + str(self.dim_one_encoder) + '_' + str(self.dim_two_encoder) + '_' \
                    + str(self.dim_three_decoder) + '-' + str(self.dim_z) + str(self.dim_two_encoder) \
                    + '_' + str(self.dim_two_decoder) + '_' +str(self.dim_three_decoder) +'.csv'
        return self.name

    def export_processed_file(self, x_full_z_y_df):
        self.file_name_output = self.get_name()
        os.chdir(self.output)
        x_full_z_y_df.to_csv(self.file_name_output, index=False)

    def __call__(self):
        return self.file_name_output

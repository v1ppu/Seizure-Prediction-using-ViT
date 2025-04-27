#create_train_test.py

from data_prep import preprocess_and_save

#test on smaller subset
edf_files = [
    'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf',
    'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_02.edf',
    'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_03.edf',
    'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_04.edf',
    'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_05.edf',
    'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_15.edf' 
]

summary_file = 'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01-summary.txt'

#train data
for edf_path in edf_files[:5]:
    preprocess_and_save(
        edf_path=edf_path,
        summary_path=summary_file,
        output_dir='data/train'
    )


#test data
preprocess_and_save(
    edf_path=edf_files[5],
    summary_path=summary_file,
    output_dir='data/test'
)


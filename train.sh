### airplane
python -u run.py --mode=train  --train_hdf5='data_hdf5/airplane_train.hdf5'   --test_hdf5='data_hdf5/airplane_test.hdf5' --domain_A=skeleton --domain_B=surface  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/airplane_train.hdf5'   --test_hdf5='data_hdf5/airplane_test.hdf5' --domain_A=skeleton --domain_B=surface  --gpu=0 --checkpoint='output_airplane_skeleton-surface/trained_models/epoch_200.ckpt'

python -u run.py --mode=train  --train_hdf5='data_hdf5/airplane_train.hdf5'   --test_hdf5='data_hdf5/airplane_test.hdf5' --domain_A=skeleton --domain_B=scan  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/airplane_train.hdf5'   --test_hdf5='data_hdf5/airplane_test.hdf5' --domain_A=skeleton --domain_B=scan  --gpu=0 --checkpoint='output_airplane_skeleton-scan/trained_models/epoch_200.ckpt'

python -u run.py --mode=train  --train_hdf5='data_hdf5/airplane_train.hdf5'   --test_hdf5='data_hdf5/airplane_test.hdf5' --domain_A=scan --domain_B=surface  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/airplane_train.hdf5'   --test_hdf5='data_hdf5/airplane_test.hdf5' --domain_A=scan --domain_B=surface  --gpu=0 --checkpoint='output_airplane_scan-surface/trained_models/epoch_200.ckpt'


### chair
python -u run.py --mode=train  --train_hdf5='data_hdf5/chair_train.hdf5'   --test_hdf5='data_hdf5/chair_test.hdf5' --domain_A=skeleton --domain_B=surface  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/chair_train.hdf5'   --test_hdf5='data_hdf5/chair_test.hdf5' --domain_A=skeleton --domain_B=surface  --gpu=0 --checkpoint='output_chair_skeleton-surface/trained_models/epoch_200.ckpt'

python -u run.py --mode=train  --train_hdf5='data_hdf5/chair_train.hdf5'   --test_hdf5='data_hdf5/chair_test.hdf5' --domain_A=skeleton --domain_B=scan  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/chair_train.hdf5'   --test_hdf5='data_hdf5/chair_test.hdf5' --domain_A=skeleton --domain_B=scan  --gpu=0 --checkpoint='output_chair_skeleton-scan/trained_models/epoch_200.ckpt'

python -u run.py --mode=train  --train_hdf5='data_hdf5/chair_train.hdf5'   --test_hdf5='data_hdf5/chair_test.hdf5' --domain_A=scan --domain_B=surface  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/chair_train.hdf5'   --test_hdf5='data_hdf5/chair_test.hdf5' --domain_A=scan --domain_B=surface  --gpu=0 --checkpoint='output_chair_scan-surface/trained_models/epoch_200.ckpt'


### bed
python -u run.py --mode=train  --train_hdf5='data_hdf5/bed_train.hdf5'   --test_hdf5='data_hdf5/bed_test.hdf5' --domain_A=cross --domain_B=surface  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/bed_train.hdf5'   --test_hdf5='data_hdf5/bed_test.hdf5' --domain_A=cross --domain_B=surface  --gpu=0 --checkpoint='output_bed_cross-surface/trained_models/epoch_200.ckpt'


### sofa
python -u run.py --mode=train  --train_hdf5='data_hdf5/sofa_train.hdf5'   --test_hdf5='data_hdf5/sofa_test.hdf5' --domain_A=cross --domain_B=surface  --gpu=0
python -u run.py --mode=test  --train_hdf5='data_hdf5/sofa_train.hdf5'   --test_hdf5='data_hdf5/sofa_test.hdf5' --domain_A=cross --domain_B=surface  --gpu=0 --checkpoint='output_sofa_cross-surface/trained_models/epoch_200.ckpt'


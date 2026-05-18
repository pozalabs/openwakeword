TRAINING_CONFIG ?= train.yml

install-dependencies:
	uv pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
	uv pip install -e .
	uv pip install -r requirements-train.txt

generate-clips:
	python openwakeword/train.py --training_config $(TRAINING_CONFIG) --generate_clips

augment-clips:
	python openwakeword/train.py --training_config $(TRAINING_CONFIG) --augment_clips

train-model:
	python openwakeword/train.py --training_config $(TRAINING_CONFIG) --train_model

train: generate-clips augment-clips train-model

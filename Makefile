install-dependencies:
	uv pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
	uv pip install -e .
	uv pip install -r requirements-train.txt

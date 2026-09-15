FROM python:3.12-slim
# No JAX, reference package, hidden tests, or repository in this image.
RUN pip install --no-cache-dir numpy==2.2.6 torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
USER 65534:65534
WORKDIR /workspace/submission
ENTRYPOINT ["python", "-I", "/runner/candidate_worker.py", "/workspace/submission/candidate.py"]

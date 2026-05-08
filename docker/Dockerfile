FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    qiskit>=1.0 \
    qiskit-aer>=0.13 \
    qiskit-ibm-runtime>=0.20 \
    numpy>=1.24

COPY src/ src/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python3", "-m"]
CMD ["quantum_agent"]

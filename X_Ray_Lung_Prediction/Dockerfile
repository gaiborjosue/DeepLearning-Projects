FROM python:3.9.5-slim

# Path: /app
ADD classification-challenge.py .

# Install tensorflow, numpy, pandas, scikit-learn, matplotlib
RUN pip install tensorflow numpy pandas scikit-learn matplotlib

CMD ["python", "classification-challenge.py"]
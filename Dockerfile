FROM apache/airflow:2.8.1

# Install all project dependencies as the airflow user.
# Baking packages into the image means no pip install at container
# startup — faster boots and no network timeout failures.
USER airflow

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    pandas==2.0.3 \
    pyarrow==14.0.2 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    shap==0.44.1
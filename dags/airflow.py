# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for the DAG
default_args = {
    'owner': 'alaska',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'Server_Workload_Clustering',
    default_args=default_args,
    description='K-Means clustering pipeline for server workload profiling',
    schedule_interval='0 6 * * *',  # Crontab: daily at 6 AM UTC
    catchup=False,
    tags=['mlops', 'clustering', 'lab1'],
)

# ---- BashOperator: Pipeline start announcement ----
start_task = BashOperator(
    task_id='start_pipeline',
    bash_command=(
        'echo "========================================" && '
        'echo "Server Workload Clustering Pipeline" && '
        'echo "Started at: $(date)" && '
        'echo "Working dir: $(pwd)" && '
        'echo "========================================" && '
        'ls -la /opt/airflow/dags/data/'
    ),
    dag=dag,
)

# ---- PythonOperator: Load data ----
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# ---- PythonOperator: Preprocess data ----
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# ---- PythonOperator: Build and save model ----
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "workload_model.sav"],
    dag=dag,
)

# ---- PythonOperator: Elbow method + predict ----
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["workload_model.sav", build_save_model_task.output],
    dag=dag,
)

# ---- BashOperator: Pipeline completion + verify outputs ----
end_task = BashOperator(
    task_id='end_pipeline',
    bash_command=(
        'echo "========================================" && '
        'echo "Pipeline completed at: $(date)" && '
        'echo "Model artifacts:" && '
        'ls -lh /opt/airflow/dags/model/ && '
        'echo "========================================" && '
        'echo "SUCCESS: Workload clustering pipeline finished."'
    ),
    dag=dag,
)

# ---- Task Dependencies ----
# start -> load -> preprocess -> build_model -> elbow -> end
#
# start_task --> load_data_task --> data_preprocessing_task -->
#   build_save_model_task --> load_model_task --> end_task

start_task >> load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> end_task

if __name__ == "__main__":
    dag.cli()
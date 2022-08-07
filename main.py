import mlflow
from dotenv import load_dotenv

import model.bert_model as m


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_dotenv()
    experiment_name = 'group9-spam-detection'
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    mlflow.tensorflow.autolog()
    with mlflow.start_run(run_id=run.info.run_id):
        bert = m.initialize_bert()
        compiled_model = m.compile_model(bert)
        x_train, x_test, y_train, y_test = m.load_train_data()
        compiled_model.fit(x_train, y_train, epochs=1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

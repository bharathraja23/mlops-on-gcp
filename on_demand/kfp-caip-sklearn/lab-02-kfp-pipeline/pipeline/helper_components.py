# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Helper components."""


from typing import NamedTuple

def get_previous_run_metric( ENDPOINT: str, experiment_id: str ) -> NamedTuple('Outputs', [('run_id', str), ('accuracy', float)]):

    import kfp as kfp
    runs_details= kfp.Client(host=ENDPOINT).list_runs(experiment_id=experiment_id, sort_by='created_at desc').to_dict()
    print(runs_details)
    latest_success_run_details=''
    print("runs_details['runs'] type {}".format(type(runs_details['runs'])))
    for i in runs_details['runs']:
        print("i['status'] type {}".format(type(i['status'])))
        if i['status'] == 'Succeeded':
            run_id=i['id']
            accuracy=i['metrics'][0]['number_value']
            break;
    print("accuracy={}".format(accuracy))
    print(type(run_id))
    return (run_id, accuracy)

def retrieve_best_run(
    project_id: str, job_id: str
) -> NamedTuple('Outputs', [('metric_value', float), ('alpha', float),
                            ('max_iter', int)]):
  """Retrieves the parameters of the best Hypertune run."""

  from googleapiclient import discovery
  from googleapiclient import errors

  ml = discovery.build('ml', 'v1')

  job_name = 'projects/{}/jobs/{}'.format(project_id, job_id)
  request = ml.projects().jobs().get(name=job_name)

  try:
    response = request.execute()
  except errors.HttpError as err:
    print(err)
  except:
    print('Unexpected error')

  print(response)

  best_trial = response['trainingOutput']['trials'][0]

  metric_value = best_trial['finalMetric']['objectiveValue']
  alpha = float(best_trial['hyperparameters']['alpha'])
  max_iter = int(best_trial['hyperparameters']['max_iter'])

  return (metric_value, alpha, max_iter)


def evaluate_model(
    dataset_path: str, model_path: str, metric_name: str
) -> NamedTuple('Outputs', [('metric_name', str), ('metric_value', float),
                            ('mlpipeline_metrics', 'Metrics')]):
  """Evaluates a trained sklearn model."""
  #import joblib
  import pickle
  import json
  import pandas as pd
  import subprocess
  import sys

  from sklearn.metrics import accuracy_score, recall_score

  df_test = pd.read_csv(dataset_path)

  X_test = df_test.drop('Cover_Type', axis=1)
  y_test = df_test['Cover_Type']

  # Copy the model from GCS
  model_filename = 'model.pkl'
  gcs_model_filepath = '{}/{}'.format(model_path, model_filename)
  print(gcs_model_filepath)
  subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, model_filename],
                        stderr=sys.stdout)

  with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

  y_hat = model.predict(X_test)

  if metric_name == 'accuracy':
    metric_value = accuracy_score(y_test, y_hat)
  elif metric_name == 'recall':
    metric_value = recall_score(y_test, y_hat)
  else:
    metric_name = 'N/A'
    metric_value = 0

  # Export the metric
  metrics = {
      'metrics': [{
          'name': metric_name,
          'numberValue': float(metric_value)
      }]
  }

  return (metric_name, metric_value, json.dumps(metrics))

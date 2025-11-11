# Import necessary libraries
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

# Load the workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(workspace=ws, model_path='model.pkl', model_name='sentiment_analysis_model')

# Deploy the model as a web service
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
service = Model.deploy(workspace=ws, name='sentiment-analysis-service', models=[model], inference_config=None, deployment_config=aci_config)
service.wait_for_deployment(show_output=True)

# Test the deployed service
print(service.scoring_uri)
import sagemaker
from sagemaker.pytorch import PyTorch

# Set up the SageMaker Estimator
estimator = PyTorch(
    entry_point='train_distill.py',
    role='SageMakerRole',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='1.9.0',
    py_version='py38',
    output_path='s3://your-bucket/path/to/output'
)

# Launch the training job
estimator.fit({'training': 's3://your-bucket/path/to/rationale_dataset.json'})

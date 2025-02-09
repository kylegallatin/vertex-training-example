gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=train1 \
  --worker-pool-spec=machine-type=ct5lp-hightpu-1t,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest,local-package-path=.,script=task.py
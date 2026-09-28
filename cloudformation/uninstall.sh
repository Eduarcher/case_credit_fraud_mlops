set -e

echo "Removing api-gateway stack" && aws cloudformation delete-stack --stack-name api-gateway && \
aws cloudformation wait stack-delete-complete --stack-name api-gateway

echo "Removing scheduler stack" && aws cloudformation delete-stack --stack-name scheduler && \
aws cloudformation wait stack-delete-complete --stack-name scheduler

echo "Removing code-pipeline-cicd stack" && aws cloudformation delete-stack --stack-name code-pipeline-cicd && \
aws cloudformation wait stack-delete-complete --stack-name code-pipeline-cicd

echo "Removing lambda-functions stack" && aws cloudformation delete-stack --stack-name lambda-functions && \
aws cloudformation wait stack-delete-complete --stack-name lambda-functions

echo "Removing ecs-stack stack" && aws cloudformation delete-stack --stack-name ecs-stack && \
aws cloudformation wait stack-delete-complete --stack-name ecs-stack

echo "Removing storage stack" && aws cloudformation delete-stack --stack-name storage && \
aws cloudformation wait stack-delete-complete --stack-name storage
echo "Uninstall complete"
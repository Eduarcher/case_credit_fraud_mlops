set -e

# Load configs as env variables using parse_yaml function
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}
export $(parse_yaml config.yml)

# Load .env file variables
set -a
source .env
set +a

# Make Temp Files folder
mkdir -p .cftmp

###########################################
############ STORAGE RESOURCES ############
###########################################
# aws cloudformation create-stack --stack-name storage \
#     --template-body file://cloudformation/templates/storage.yaml
# aws cloudformation wait stack-create-complete --stack-name storage

###########################################
########### ECS TASK DEFINITION ###########
###########################################
# Upload container environment
ENV_FILE_S3_PATH=s3://${AWS_SAGEMAKER_S3_BUCKET_NAME}/${AWS_SAGEMAKER_S3_BUCKET_NAME_FOLDER_PREFIX}/environment/
ENV_FILE_S3_ARN=arn:aws:s3:::${AWS_SAGEMAKER_S3_BUCKET_NAME}/${AWS_SAGEMAKER_S3_BUCKET_NAME_FOLDER_PREFIX}/environment/.env
aws s3 cp .env $ENV_FILE_S3_PATH
   
# Deploy stack
aws cloudformation create-stack --stack-name ecs-stack \
    --template-body file://cloudformation/templates/ecs.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters ParameterKey=ECSTaskDefinitionName,ParameterValue=${ECS_ECSTaskDefinitionName} \
    ParameterKey=ContainerEnvironmentFileS3ARN,ParameterValue=${ENV_FILE_S3_ARN} && \
aws cloudformation wait stack-create-complete --stack-name ecs-stack

###########################################
####### LAMBDA FUNCTIONS AND LAYERS #######
###########################################
# Upload Lambda Functions zipped content
mkdir -p .cftmp/lambda_functions && \
zip -j -r .cftmp/lambda_functions/lambda_run_pipeline.zip cloudformation/src/lambda/run_pipeline && \
zip -j -r .cftmp/lambda_functions/lambda_register_model.zip cloudformation/src/lambda/register_model && \
zip -j -r .cftmp/lambda_functions/lambda_deploy_model.zip cloudformation/src/lambda/deploy_model && \
zip -j -r .cftmp/lambda_functions/lambda_route_health_model.zip cloudformation/src/lambda/route_health && \
zip -j -r .cftmp/lambda_functions/lambda_route_inference_model.zip cloudformation/src/lambda/route_inference && \
aws s3 cp .cftmp/lambda_functions \
    s3://${AWS_SAGEMAKER_S3_BUCKET_NAME}/${AWS_SAGEMAKER_S3_BUCKET_NAME_FOLDER_PREFIX}/lambda_functions/ \
    --recursive

# Upload Lambda Layers zipped content
mkdir -p .cftmp/lambda_layers && \
TARGET=$(pwd) && (cd cloudformation/src/layers/lambda_layer_mlflow_skinny && \
    zip -r ${TARGET}/.cftmp/lambda_layers/lambda_layer_mlflow_skinny.zip .) && \
aws s3 cp .cftmp/lambda_layers \
    s3://${AWS_SAGEMAKER_S3_BUCKET_NAME}/${AWS_SAGEMAKER_S3_BUCKET_NAME_FOLDER_PREFIX}/lambda_layers/ \
    --recursive

# Deploy stack
aws cloudformation create-stack --stack-name lambda-functions \
    --template-body file://cloudformation/templates/lambda.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters ParameterKey=S3BucketName,ParameterValue=${AWS_SAGEMAKER_S3_BUCKET_NAME} \
    ParameterKey=LambdaRunPipelineFunctionName,ParameterValue=${ECS_RunPipelineLambdaFunctionName} \
    ParameterKey=LambdaRegisterModelFunctionName,ParameterValue=${Registry_RegisterModelLambdaFunctionName} \
    ParameterKey=LambdaDeployModelFunctionName,ParameterValue=${Deployment_DeployLambdaFunctionName} \
    ParameterKey=LambdaRouteHealthModelFunctionName,ParameterValue=${APIGateway_InferenceHealthLambdaFunctionName} \
    ParameterKey=LambdaRouteInferenceModelFunctionName,ParameterValue=${APIGateway_InferenceEndpointLambdaFunctionName} \
    ParameterKey=EndpointName,ParameterValue=${Deployment_EndpointName} \
    ParameterKey=RunPipelineECSTaskDefinitionName,ParameterValue=${ECS_ECSTaskDefinitionName} \
    ParameterKey=RunPipelineVPCID,ParameterValue=${VPC_ID} \
    ParameterKey=DeployModelMinCapacity,ParameterValue=${Deployment_DeployModelMinCapacity} \
    ParameterKey=DeployModelMaxCapacity,ParameterValue=${Deployment_DeployModelMaxCapacity}

aws cloudformation wait stack-create-complete --stack-name lambda-functions

##########################################
########## CODE PIPELINE CI/CD ###########
##########################################
aws cloudformation create-stack --stack-name code-pipeline-cicd \
    --template-body file://cloudformation/templates/code_pipeline.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters ParameterKey=GitHubConnectionArn,ParameterValue=${GITHUB_CONNECTION_ARN} \
    ParameterKey=GitHubRepositoryName,ParameterValue=${GITHUB_REPOSITORY_NAME} \
    ParameterKey=MainBranchName,ParameterValue=${MAIN_BRANCH_NAME} \
    ParameterKey=RunPipelineLambdaFunctionName,ParameterValue=${ECS_RunPipelineLambdaFunctionName}

aws cloudformation wait stack-create-complete --stack-name code-pipeline-cicd

# ###########################################
# ################ SCHEDULER ################
# ###########################################
VPC_SUBNETS=$(aws ec2 describe-subnets --filter Name=vpc-id,Values=${VPC_ID} \
    --query 'Subnets[?State==`available`].SubnetId'  --output text | tr '\t' ',')
aws cloudformation create-stack --stack-name scheduler \
    --template-body file://cloudformation/templates/scheduler.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters ParameterKey=RunPipelineECSSubnetList,ParameterValue=\"${VPC_SUBNETS}\" \
    ParameterKey=ECSTaskDefinitionName,ParameterValue=${ECS_ECSTaskDefinitionName} \
    ParameterKey=SchedulerCronExpression,ParameterValue="${CRON_SCHEDULE}"

aws cloudformation wait stack-create-complete --stack-name scheduler

# ###########################################
# ############### API GATEWAY ###############
# ###########################################
aws cloudformation create-stack --stack-name api-gateway \
    --template-body file://cloudformation/templates/apigateway.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters ParameterKey=LambdaRouteHealthModelFunctionName,ParameterValue=${APIGateway_InferenceHealthLambdaFunctionName}

aws cloudformation wait stack-create-complete --stack-name api-gateway

## Clean temp files
rm -rf .cftmp

echo "Install complete"

# AWS setup for this repo

## 1) GitHub secrets
Create these repository secrets in GitHub:

- CLOUD_CREDENTIALS
  ```json
  {
    "aws_access_key_id": "YOUR_ACCESS_KEY",
    "aws_secret_access_key": "YOUR_SECRET_KEY"
  }
  ```
- AWS_REGION
  - your AWS region, e.g. `us-east-1`
- CLOUD_BUCKET
  - your S3 bucket name, e.g. `my-mlops-bucket`
- VM_HOST
- VM_USER
- VM_SSH_KEY

## 2) Local DVC remote
Run these commands locally:

```bash
pip install -r requirements.txt

# initialize DVC if needed
dvc init

dvc remote add -d myremote s3://<YOUR_BUCKET>/dvc

dvc remote modify myremote access_key_id <YOUR_ACCESS_KEY>
dvc remote modify myremote secret_access_key <YOUR_SECRET_KEY>
dvc remote modify myremote region <YOUR_REGION>
```

Then track and push data:

```bash
python generate_data.py
dvc add data/train_phase1.csv data/eval.csv data/train_phase2.csv
git add .dvc/config data/*.dvc .gitignore
git commit -m "feat: track datasets with dvc"
dvc push
```

## 3) VM setup
On your EC2 instance:

```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install fastapi uvicorn scikit-learn joblib boto3
mkdir -p ~/models ~/src
```

Copy your AWS credentials file to the VM if needed and make sure the service uses the environment variables.

## 4) Push to GitHub
```bash
git add .
git commit -m "feat: add AWS CI/CD pipeline"
git push origin main
```

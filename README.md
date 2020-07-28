# Running ktrain

Build docker container:

```docker build -t run_bert <project_path>```

Run docker container:

```docker run --rm --name run_bert -v <project_path>/:/run_ktrain -i -t run_bert /bin/bash```

Execute:

```bash run_ktrain.sh <dataset_path> <results_path>```

For more information about building and running a docker container, see: https://docs.docker.com/

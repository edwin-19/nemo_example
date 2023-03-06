docker run --network host -it -v /home/revenuemonster/Documents/Edwin/webinar/triton_models:/models nvcr.io/nvidia/tritonserver:21.09-py3 /bin/bash
# tritonserver --model-repository=/models
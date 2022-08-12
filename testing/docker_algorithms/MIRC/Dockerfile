FROM tensorflow/tensorflow:2.7.0-gpu

RUN rm -f /etc/apt/sources.list.d/cuda.list && rm -f /etc/apt/sources.list.d/nvidia-ml.list && apt-get update && apt-get install -y git
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /opt/algorithm /input /output /dvn_models && chown algorithm:algorithm /opt/algorithm /input /output /dvn_models

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm /dvn_models/ /dvn_models/

RUN python -m pip install --user -U pip
RUN python -m pip install --user evalutils==0.3.1 && python -m pip uninstall -y SimpleITK
RUN python -m pip install --user -r requirements.txt

ENTRYPOINT python -m process $0 $@
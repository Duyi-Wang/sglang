FROM henryx/xsgl:v0.5.6-rocm711-mi35x-test-20251205

# For mori and ainic dependencies
RUN apt-get install -y initramfs-tools jq libopenmpi-dev libpci-dev

ENV MORI_GPU_ARCHS=gfx942;gfx950
ARG MORI_COMMIT="ionic_new_950_1128"
ARG MORI_REPO="https://github.com/ROCm/mori.git"


ARG AITER_COMMIT="mla_fake_non_persistent"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"

ARG SGL_REPO="https://github.com/Duyi-Wang/sglang.git"
ARG SGL_BRANCH="mori_ep_1205"

RUN update-alternatives --install /usr/bin/pip pip /usr/local/lib/python3.12/dist-packages/pip 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/lib/python3.12/dist-packages/pip 1

WORKDIR /sgl-workspace

RUN pip uninstall -y mori
RUN git clone ${MORI_REPO} \
    && cd mori \
    && git checkout ${MORI_COMMIT} \
    && git submodule update --init --recursive

# AITER
RUN pip uninstall -y aiter
RUN rm -rf aiter
RUN git clone ${AITER_REPO} aiter
RUN cd aiter \
    && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py develop

# SGLang
RUN pip uninstall -y sglang
RUN cd sglang \
    && git restore . \
    && git remote add duyi ${SGL_REPO} \
    && git fetch duyi \
    && git checkout ${SGL_BRANCH} \
    && cd sgl-kernel \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=$GPU_ARCH_LIST python setup_rocm.py install \
    && cd .. \
    && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml \
    && python -m pip --no-cache-dir install -e "python[all_hip]" ${NO_DEPS_FLAG}

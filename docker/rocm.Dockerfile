# Usage (to build SGLang ROCm docker image):
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx942 -t v0.5.9-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx942-rocm720 -t v0.5.9-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx950 -t v0.5.9-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx950-rocm720 -t v0.5.9-rocm720-mi35x -f rocm.Dockerfile .

# Usage (to build SGLang ROCm + Mori docker image):
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx942 --build-arg ENABLE_MORI=1 --build-arg NIC_BACKEND=ainic -t v0.5.9-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx942-rocm720 --build-arg ENABLE_MORI=1 --build-arg NIC_BACKEND=ainic -t v0.5.9-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx950 --build-arg ENABLE_MORI=1 --build-arg NIC_BACKEND=ainic -t v0.5.9-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.9 --build-arg GPU_ARCH=gfx950-rocm720 --build-arg ENABLE_MORI=1 --build-arg NIC_BACKEND=ainic -t v0.5.9-rocm720-mi35x -f rocm.Dockerfile .

# Default base images
ARG BASE_IMAGE_942="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_942_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
ARG BASE_IMAGE_950="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_950_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"

# This is necessary for scope purpose
ARG GPU_ARCH=gfx950

# ===============================
# Base image 942 with rocm700 and args
FROM $BASE_IMAGE_942 AS gfx942
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="v0.1.11.post1"

# ===============================
# Base image 942 with rocm720 and args
FROM $BASE_IMAGE_942_ROCM720 AS gfx942-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="v0.1.11.post1"

# ===============================
# Base image 950 and args
FROM $BASE_IMAGE_950 AS gfx950
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="v0.1.11.post1"

# ===============================
# Base image 950 with rocm720 and args
FROM $BASE_IMAGE_950_ROCM720 AS gfx950-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="v0.1.11.post1"

# ================================================================
# Builder stage: sgl-model-gateway (builds Rust wheel, discarded)
# ================================================================
FROM ${GPU_ARCH} AS builder-gateway

ARG GPU_ARCH=gfx950
ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}

ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && pip install --no-cache-dir maturin \
    && (git clone --depth 1 -b ${SGL_BRANCH} ${SGL_REPO} /tmp/sglang 2>/dev/null \
        || (git clone ${SGL_REPO} /tmp/sglang && cd /tmp/sglang && git checkout ${SGL_BRANCH})) \
    && cd /tmp/sglang/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && CARGO_BUILD_JOBS=4 maturin build --release --features vendored-openssl --out /tmp/gateway-wheel

# ================================================================
# Builder stage: Mooncake (builds C++ lib, discarded)
# ================================================================
FROM ${GPU_ARCH} AS builder-mooncake

ARG GPU_ARCH=gfx950
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ARG MOONCAKE_COMMIT="b6a841dc78c707ec655a563453277d969fb8f38d"

ENV PATH=$PATH:/usr/local/go/bin

# Build Mooncake and install to a staging directory for later COPY
# Always create the staging dir so COPY --from=builder-mooncake never fails
RUN mkdir -p /mooncake-install/usr/local \
    && apt-get update && apt-get install -y --no-install-recommends \
        zip unzip wget gcc make libtool autoconf \
        librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
        libibverbs-dev rdma-core \
        openssh-server openmpi-bin openmpi-common libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && (git clone --depth 1 -b ${MOONCAKE_COMMIT} ${MOONCAKE_REPO} /tmp/Mooncake 2>/dev/null \
        || (git clone ${MOONCAKE_REPO} /tmp/Mooncake && cd /tmp/Mooncake && git checkout ${MOONCAKE_COMMIT})) \
    && cd /tmp/Mooncake \
    && git submodule update --init --recursive \
    && bash dependencies.sh -y \
    && rm -rf /usr/local/go \
    && wget -q https://go.dev/dl/go1.22.2.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz \
    && rm go1.22.2.linux-amd64.tar.gz \
    && mkdir -p build && cd build \
    && cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON \
    && make -j "$(nproc)" \
    && DESTDIR=/mooncake-install make install

# ================================================================
# Builder stage: FHT (builds HIP extension, discarded)
# ================================================================
FROM ${GPU_ARCH} AS builder-fht

ARG FHT_REPO="https://github.com/jeffdaily/fast-hadamard-transform.git"
ARG FHT_BRANCH="rocm"
ARG FHT_COMMIT="46efb7d776d38638fc39f3c803eaee3dd7016bd1"

RUN pip install --no-cache-dir wheel \
    && (git clone --depth 1 -b "${FHT_COMMIT}" "${FHT_REPO}" /tmp/fht 2>/dev/null \
        || (git clone "${FHT_REPO}" /tmp/fht && cd /tmp/fht && git checkout -f "${FHT_COMMIT}")) \
    && cd /tmp/fht \
    && FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE python setup.py bdist_wheel -d /tmp/fht-wheel

# ================================================================
# Builder stage: TileLang (builds wheel, discarded)
# ================================================================
FROM ${GPU_ARCH} AS builder-tilelang

ARG GPU_ARCH=gfx950
ARG TILELANG_REPO="https://github.com/tile-ai/tilelang.git"
ARG TILELANG_COMMIT="a55a82302bf7f3c5af635b5c9146f728185cc900"

ENV DEBIAN_FRONTEND=noninteractive

RUN /bin/bash -lc 'set -euo pipefail; \
  echo "[TileLang] Building TileLang wheel for ${GPU_ARCH}"; \
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget curl ca-certificates gnupg \
      libgtest-dev libgmock-dev \
      libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev \
      python3 python3-dev python3-setuptools python3-pip python3-apt \
      gcc libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
      cmake ninja-build pkg-config libstdc++6 software-properties-common \
  && rm -rf /var/lib/apt/lists/*; \
  \
  VENV_PY="/opt/venv/bin/python"; \
  VENV_PIP="/opt/venv/bin/pip"; \
  if [ ! -x "$VENV_PY" ]; then VENV_PY="python3"; fi; \
  if [ ! -x "$VENV_PIP" ]; then VENV_PIP="pip3"; fi; \
  \
  cmake -S /usr/src/googletest -B /tmp/build-gtest -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build /tmp/build-gtest -j"$(nproc)" && \
  cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/ && \
  rm -rf /tmp/build-gtest; \
  \
  "$VENV_PIP" install --no-cache-dir --upgrade "setuptools>=77.0.3,<80" wheel cmake ninja scikit-build-core; \
  \
  LLVM_CONFIG_PATH=""; \
  for p in /opt/rocm/llvm/bin/llvm-config /opt/rocm/llvm-*/bin/llvm-config /opt/rocm-*/llvm*/bin/llvm-config; do \
    if [ -x "$p" ]; then LLVM_CONFIG_PATH="$p"; break; fi; \
  done; \
  if [ -z "$LLVM_CONFIG_PATH" ]; then \
    echo "[TileLang] ROCm llvm-config not found; installing LLVM 18..."; \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" > /etc/apt/sources.list.d/llvm.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends llvm-18; \
    rm -rf /var/lib/apt/lists/*; \
    LLVM_CONFIG_PATH="$(command -v llvm-config-18)"; \
    if [ -z "$LLVM_CONFIG_PATH" ]; then echo "ERROR: llvm-config-18 not found after install"; exit 1; fi; \
  fi; \
  echo "[TileLang] Using LLVM_CONFIG at: $LLVM_CONFIG_PATH"; \
  export PATH="$(dirname "$LLVM_CONFIG_PATH"):/usr/local/bin:${PATH}"; \
  export LLVM_CONFIG="$LLVM_CONFIG_PATH"; \
  \
  mkdir -p /usr/local/bin && \
  printf "#!/usr/bin/env bash\nexec \"%s\" \"\$@\"\n" "$LLVM_CONFIG_PATH" > /usr/local/bin/llvm-config-16 && \
  chmod +x /usr/local/bin/llvm-config-16; \
  \
  "$VENV_PIP" install --no-cache-dir "cython>=0.29.36,<3.0" "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" "z3-solver==4.15.4.0"; \
  \
  (git clone --depth 1 --recursive -b "${TILELANG_COMMIT}" "${TILELANG_REPO}" /opt/tilelang 2>/dev/null \
    || (git clone --recursive "${TILELANG_REPO}" /opt/tilelang && cd /opt/tilelang && git checkout -f "${TILELANG_COMMIT}")) && \
  cd /opt/tilelang && \
  git submodule update --init --recursive && \
  if [ -f pyproject.toml ]; then sed -i "/^[[:space:]]*\"torch/d" pyproject.toml || true; fi && \
  export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG} -DSKBUILD_SABI_VERSION= ${CMAKE_ARGS:-}" && \
  "$VENV_PIP" wheel -w /tmp/tilelang-wheel . -v --no-build-isolation --no-deps && \
  "$VENV_PIP" cache purge || true'

# ================================================================
# Builder stage: Triton (builds wheel, discarded)
# ================================================================
FROM ${GPU_ARCH} AS builder-triton

ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG TRITON_COMMIT="42270451990532c67e69d753fbd026f28fcc4840"

# BUILD_TRITON is inherited as ENV from the selected base stage
# Always create output dir so COPY --from never fails
RUN mkdir -p /tmp/triton-wheel \
    && if [ "$BUILD_TRITON" = "1" ]; then \
        pip uninstall -y triton \
     && apt-get update && apt-get install -y --no-install-recommends cmake \
     && rm -rf /var/lib/apt/lists/* \
     && (git clone --depth 1 -b ${TRITON_COMMIT} ${TRITON_REPO} /tmp/triton-custom 2>/dev/null \
         || (git clone ${TRITON_REPO} /tmp/triton-custom && cd /tmp/triton-custom && git checkout ${TRITON_COMMIT})) \
     && cd /tmp/triton-custom \
     && pip install --no-cache-dir -r python/requirements.txt \
     && pip wheel --no-cache-dir --no-deps -w /tmp/triton-wheel .; \
    fi

# ================================================================
# Final stage
# ================================================================
FROM ${GPU_ARCH}

# This is necessary for scope purpose, again
ARG GPU_ARCH=gfx950
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}
ENV PYTORCH_ROCM_ARCH=gfx942;gfx950

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}

# Version override for setuptools_scm (used in nightly builds)
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""

ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT=""
ENV AITER_COMMIT="${AITER_COMMIT:-${AITER_COMMIT_DEFAULT}}"

ARG LLVM_REPO="https://github.com/jrbyrnes/llvm-project.git"
ARG LLVM_BRANCH="MainOpSelV2"
ARG LLVM_COMMIT="6520ace8227ffe2728148d5f3b9872a870b0a560"

ARG ENABLE_MORI=0
ARG NIC_BACKEND=none

ARG MORI_REPO="https://github.com/ROCm/mori.git"
ARG MORI_COMMIT="v0.1.0"

# AMD AINIC apt repo settings
ARG AINIC_VERSION=1.117.5
ARG UBUNTU_CODENAME=jammy
USER root

# Fix hipDeviceGetName returning empty string in ROCm 7.0 docker images.
# The ROCm 7.0 base image is missing libdrm-amdgpu-common which provides the
# amdgpu.ids device-ID-to-marketing-name mapping file.
# ROCm 7.2 base images already ship these packages, so this step is skipped.
# See https://github.com/ROCm/ROCm/issues/5992
RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm720*) \
        echo "ROCm 7.2 (GPU_ARCH=${GPU_ARCH}): libdrm-amdgpu packages already present, skipping"; \
        ;; \
      *) \
        echo "ROCm 7.0 (GPU_ARCH=${GPU_ARCH}): installing libdrm-amdgpu packages"; \
        curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
          | gpg --dearmor -o /etc/apt/keyrings/amdgpu-graphics.gpg \
        && echo 'deb [arch=amd64,i386 signed-by=/etc/apt/keyrings/amdgpu-graphics.gpg] https://repo.radeon.com/graphics/7.0/ubuntu jammy main' \
          > /etc/apt/sources.list.d/amdgpu-graphics.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
             libdrm-amdgpu-common \
             libdrm-amdgpu-amdgpu1 \
             libdrm2-amdgpu \
        && rm -rf /var/lib/apt/lists/* \
        && cp /opt/amdgpu/share/libdrm/amdgpu.ids /usr/share/libdrm/amdgpu.ids; \
        ;; \
    esac

# Install some basic utilities + pip deps (combined to reduce layers)
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir setuptools_scm IPython orjson python-multipart torchao==0.9.0 pybind11 \
    && apt-get purge -y sccache 2>/dev/null || true; python -m pip uninstall -y sccache 2>/dev/null || true; rm -f "$(which sccache 2>/dev/null)" || true

# Install AMD SMI Python package from ROCm distribution.
# The ROCm 7.2 base image (rocm/pytorch) does not pre-install this package.
RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm720*) \
        echo "ROCm 7.2 flavor detected from GPU_ARCH=${GPU_ARCH}"; \
        cd /opt/rocm/share/amd_smi \
        && python3 -m pip install --no-cache-dir . \
        ;; \
      *) \
        echo "Not rocm720 (GPU_ARCH=${GPU_ARCH}), skip amdsmi installation"; \
        ;; \
    esac

WORKDIR /sgl-workspace

# -----------------------
# llvm
RUN if [ "$BUILD_LLVM" = "1" ]; then \
     ENV HIP_CLANG_PATH="/sgl-workspace/llvm-project/build/bin/" \
     git clone ${LLVM_REPO} \
     && cd llvm-project \
     && git checkout ${LLVM_COMMIT} \
     && git fetch --depth 1 origin $(git rev-parse HEAD) 2>/dev/null || true \
     && git reflog expire --expire=now --all && git gc --prune=now \
     && mkdir build \
     && cd build \
     && cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;" -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm \
     && make -j$(nproc) \
     && find /sgl-workspace/llvm-project -name '*.o' -delete; \
    fi

# -----------------------
# AITER
# Unset setuptools_scm override so AITER gets its own version (AITER_COMMIT), not SGLang's
# (SETUPTOOLS_SCM_PRETEND_VERSION is set later for SGLang nightly builds and would otherwise
# leak into AITER's version when AITER uses setuptools_scm)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=
RUN pip uninstall -y aiter 2>/dev/null || true \
 && pip install --no-cache-dir flydsl==0.0.1.dev95158637 psutil pybind11 \
 && git clone ${AITER_REPO} \
 && cd aiter \
 && git checkout ${AITER_COMMIT} \
 && git submodule update --init --recursive \
 && git fetch --depth 1 origin $(git rev-parse HEAD) 2>/dev/null || true \
 && git reflog expire --expire=now --all && git gc --prune=now \
 # Hot patches for AITER (ROCm 7.2 only)
 && case "${GPU_ARCH}" in \
      *rocm720*) \
        echo "ROCm 7.2 flavor detected from GPU_ARCH=${GPU_ARCH}"; \
        sed -i '459 s/if.*:/if False:/' aiter/ops/triton/attention/pa_mqa_logits.py; \
        ;; \
      *) \
        echo "Not rocm720 (GPU_ARCH=${GPU_ARCH}), skip patch"; \
        ;; \
    esac \
 # [WA] from kk-huang - aiter triton gemm config issue (ROCm/aiter#2173)
 && echo "[AITER] GPU_ARCH=${GPU_ARCH}" \
 && sed -i '/c1 = torch.empty((M, D, S1 + S3), dtype=dtype, device=x.device)/i\    config = dict(config)' aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py \
 && if [ "$BUILD_AITER_ALL" = "1" ] && [ "$BUILD_LLVM" = "1" ]; then \
      sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py build_ext --inplace" \
      && sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ GPU_ARCHS=$GPU_ARCH_LIST pip install --no-cache-dir -e ."; \
    elif [ "$BUILD_AITER_ALL" = "1" ]; then \
      sh -c "PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py build_ext --inplace" \
      && sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --no-cache-dir -e ."; \
    else \
      sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --no-cache-dir -e ."; \
    fi \
 && echo "export PYTHONPATH=/sgl-workspace/aiter:\${PYTHONPATH}" >> /etc/bash.bashrc \
 # Cleanup: object files (editable install keeps source; keep .git for dev)
 && find /sgl-workspace/aiter -name '*.o' -delete 2>/dev/null || true

# -----------------------
# Install Mooncake from builder stage
# Runtime apt deps are needed (RDMA/openmpi); built libs come from builder-mooncake

RUN if [ "$BUILD_MOONCAKE" = "1" ]; then \
     apt-get update && apt-get install -y --no-install-recommends \
         librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
         libibverbs-dev rdma-core \
         openssh-server openmpi-bin openmpi-common libopenmpi-dev \
         libgoogle-glog-dev libjsoncpp-dev libunwind-dev libnuma-dev \
         libboost-all-dev libssl-dev libyaml-cpp-dev libgflags-dev \
         libgrpc-dev libgrpc++-dev libprotobuf-dev \
     && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy Mooncake built artifacts from builder stage (only if BUILD_MOONCAKE=1)
# COPY is unconditional in Dockerfile, so the files always get copied;
# they are small (libs only) when mooncake was not built.
# /usr/local/ contains libetcd_wrapper.so, transfer_engine_c.h, mooncake_master
# The Python mooncake package is installed to site-packages via absolute path
COPY --from=builder-mooncake /mooncake-install/ /

# -----------------------
# Build SGLang (sgl-kernel non-editable + sglang editable in single layer)
ARG BUILD_TYPE=all

# Set version for setuptools_scm if provided (for nightly builds). Only pass in the SGLang
# pip install RUN so it does not affect AITER, sgl-model-gateway, TileLang, FHT, MORI, etc.
ARG SETUPTOOLS_SCM_PRETEND_VERSION

RUN pip uninstall -y sgl_kernel sglang 2>/dev/null || true \
    && git clone ${SGL_REPO} \
    && cd sglang \
    && git checkout ${SGL_BRANCH} \
    && git fetch --depth 1 origin $(git rev-parse HEAD) 2>/dev/null || true \
    && git reflog expire --expire=now --all && git gc --prune=now \
    && cd sgl-kernel \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=$GPU_ARCH_LIST python setup_rocm.py install \
    && cd .. \
    && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml \
    && if [ "$BUILD_TYPE" = "srt" ]; then \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install -e "python[srt_hip,diffusion_hip]"; \
       else \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install -e "python[all_hip]"; \
       fi \
    && find . -name '*.o' -delete 2>/dev/null || true \
    && python -m pip cache purge

# Copy config files to support MI300X in virtualized environments (MI300X_VF).  Symlinks will not be created in image build.
RUN find /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/ \
         /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/ \
         -type f -name '*MI300X*' | xargs -I {} sh -c 'vf_config=$(echo "$1" | sed "s/MI300X/MI300X_VF/"); cp "$1" "$vf_config"' -- {}

# -----------------------
# Install sgl-model-gateway wheel from builder stage (Rust never enters final image)
COPY --from=builder-gateway /tmp/gateway-wheel/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --force-reinstall /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# -----------------------
# TileLang (non-editable, installed from builder wheel)
# Runtime deps: tvm_ffi is required by the bundled TVM; z3-solver by TVM's solver
COPY --from=builder-tilelang /tmp/tilelang-wheel/*.whl /tmp/wheels/
RUN /bin/bash -lc 'set -euo pipefail; \
  VENV_PIP="/opt/venv/bin/pip"; \
  if [ ! -x "$VENV_PIP" ]; then VENV_PIP="pip3"; fi; \
  "$VENV_PIP" install --no-cache-dir \
    "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" \
    "z3-solver==4.15.4.0" \
  && "$VENV_PIP" install --no-cache-dir --no-deps --force-reinstall /tmp/wheels/*.whl \
  && rm -rf /tmp/wheels; \
  VENV_PY="/opt/venv/bin/python"; \
  if [ ! -x "$VENV_PY" ]; then VENV_PY="python3"; fi; \
  "$VENV_PY" -c "import tilelang; print(tilelang.__version__)"'

# -----------------------
# Install FHT wheel from builder stage (source never enters final image)
COPY --from=builder-fht /tmp/fht-wheel/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --force-reinstall /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# -----------------------
# Python tools
RUN python3 -m pip install --no-cache-dir \
    py-spy \
    tabulate

# -----------------------
# MORI (optional)
RUN /bin/bash -lc 'set -euo pipefail; \
  if [ "${ENABLE_MORI}" != "1" ]; then \
    echo "[MORI] Skipping (ENABLE_MORI=${ENABLE_MORI})"; \
    exit 0; \
  fi; \
  echo "[MORI] Enabling MORI (NIC_BACKEND=${NIC_BACKEND})"; \
  \
  # Base deps for MORI build
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      g++ \
      jq \
      libopenmpi-dev \
      libpci-dev \
      initramfs-tools \
  && rm -rf /var/lib/apt/lists/*; \
  \
  # NIC backend deps
  case "${NIC_BACKEND}" in \
    # default: mlx5
    none) \
      export USE_IONIC="OFF"; \
      export USE_BNXT="OFF"; \
      ;; \
    # AMD NIC
    ainic) \
      export USE_IONIC="ON"; \
      export USE_BNXT="OFF"; \
      apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg apt-transport-https && \
      rm -rf /var/lib/apt/lists/* && mkdir -p /etc/apt/keyrings; \
      curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg; \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/amdainic.list; \
      apt-get update && apt-get install -y --no-install-recommends \
          libionic-dev \
          ionic-common \
      ; \
      rm -rf /var/lib/apt/lists/*; \
      ;; \
    *) \
      echo "ERROR: unknown NIC_BACKEND=${NIC_BACKEND}. Use one of: none, ainic"; \
      exit 2; \
      ;; \
  esac; \
  \
  # Build/install MORI
  export MORI_GPU_ARCHS="${GPU_ARCH_LIST}"; \
  echo "[MORI] MORI_GPU_ARCHS=${MORI_GPU_ARCHS} USE_IONIC=${USE_IONIC} USE_BNXT=${USE_BNXT}"; \
  rm -rf /sgl-workspace/mori; \
  git clone "${MORI_REPO}" /sgl-workspace/mori; \
  cd /sgl-workspace/mori; \
  git checkout "${MORI_COMMIT}"; \
  git submodule update --init --recursive; \
  git fetch --depth 1 origin $(git rev-parse HEAD) 2>/dev/null || true; \
  git reflog expire --expire=now --all && git gc --prune=now; \
  python3 setup.py develop; \
  python3 -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))" > /etc/ld.so.conf.d/torch.conf; \
  ldconfig; \
  echo "export PYTHONPATH=/sgl-workspace/mori:\${PYTHONPATH}" >> /etc/bash.bashrc; \
  find /sgl-workspace/mori -name "*.o" -delete 2>/dev/null || true; \
  echo "[MORI] Done."'

# -----------------------
# Hot patch: torch-ROCm
# The artifact hardcoded the supported triton version to be 3.5.1.
# Rewrite the restriction directly.
# Combined into single layer to avoid the 3.3 GB extracted wheel lingering in a prior layer
ARG TORCH_ROCM_FILE="torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl"
RUN mkdir -p /tmp/whl && cat > /tmp/whl/hack.py <<"PY"
import zipfile, csv, os, re
from pathlib import Path

fname = os.environ["TORCH_ROCM_FILE"]
in_whl  = Path("/")   / fname
out_whl = Path("/tmp")/ fname
work = Path("/tmp/whl")

# 1) Extract
with zipfile.ZipFile(in_whl, "r") as z:
    z.extractall(work)

# 2) Locate dist-info and patch METADATA (edit this logic to match your exact line)
dist_info = next(work.glob("*.dist-info"))
meta = dist_info / "METADATA"
txt = meta.read_text(encoding="utf-8")

# Example: replace one exact requirement form.
# Adjust the string to match what you actually see.
pat = r"^Requires-Dist:\s*triton==3.5.1[^\s]*;"
txt2, n = re.subn(pat, r"triton>=3.5.1;", txt, flags=re.MULTILINE)
if txt2 == txt:
    raise SystemExit("Did not find expected Requires-Dist line to replace in METADATA")
meta.write_text(txt2, encoding="utf-8")

# 3) Hacky step: blank hash/size columns in RECORD
record = dist_info / "RECORD"
rows = []
with record.open(newline="", encoding="utf-8") as f:
    for r in csv.reader(f):
        if not r:
            continue
        # keep filename, blank out hash and size
        rows.append([r[0], "", ""])
with record.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# 4) Re-zip as a wheel
with zipfile.ZipFile(out_whl, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in work.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(work).as_posix())

print("Wrote", out_whl)
PY

RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm720*) \
        echo "ROCm 7.2 flavor detected from GPU_ARCH=${GPU_ARCH}"; \
        cd /tmp/whl \
        && export TORCH_ROCM_FILE="${TORCH_ROCM_FILE}" \
        && python hack.py \
        && python3 -m pip install --force --no-deps /tmp/${TORCH_ROCM_FILE} \
        && rm -rf /tmp/whl /tmp/${TORCH_ROCM_FILE}; \
        ;; \
      *) \
        echo "Not rocm720 (GPU_ARCH=${GPU_ARCH}), skip patch"; \
        rm -rf /tmp/whl; \
        ;; \
    esac


# -----------------------
# Triton (non-editable, installed from builder wheel)
# For ROCm 7.2, this custom build breaks pip dependency management,
# so future `pip install` will break the ROCm stack.
# A workaround for this is to reinstall the default triton
# wheel with the `rocm/pytorch` image in the root directory.
COPY --from=builder-triton /tmp/triton-wheel/ /tmp/triton-wheel/
RUN set -eux; \
    if ls /tmp/triton-wheel/*.whl 1>/dev/null 2>&1; then \
        pip uninstall -y triton \
     && pip install --no-cache-dir --force-reinstall /tmp/triton-wheel/*.whl; \
    fi; \
    rm -rf /tmp/triton-wheel

# -----------------------
# Performance environment variable.

# Skip CuDNN compatibility check - not applicable for ROCm (uses MIOpen instead)
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_INT4_WEIGHT=0
ENV SGLANG_MOE_PADDING=1
ENV SGLANG_ROCM_DISABLE_LINEARQUANT=0
ENV SGLANG_ROCM_FUSED_DECODE_MLA=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_USE_ROCM700A=1

ENV NCCL_MIN_NCHANNELS=112
ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8
ENV TORCHINDUCTOR_MAX_AUTOTUNE=1
ENV TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

CMD ["/bin/bash"]

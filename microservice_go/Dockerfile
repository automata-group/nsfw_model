# Stage 1: Build stage
FROM golang:1.22-bookworm AS builder
WORKDIR /app

# Download dependencies
COPY go.mod ./
# RUN go mod tidy

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-w -s" -o moderation-server .

# Stage 2: Distroless / Debian-Slim lightweight production image
FROM debian:bookworm-slim
WORKDIR /app

# Install ONNX Runtime shared library
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates curl \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz \
    && tar -xvzf onnxruntime-linux-x64-1.17.3.tgz \
    && cp onnxruntime-linux-x64-1.17.3/lib/libonnxruntime.so* /usr/lib/ \
    && rm -rf onnxruntime* \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/moderation-server .
COPY nsfw_lcnet050.onnx .
COPY multimodal_anatomy_weights.json .

ENV PORT=8080
ENV MODEL_PATH=nsfw_lcnet050.onnx
ENV ORT_LIB_PATH=/usr/lib/libonnxruntime.so

EXPOSE 8080
CMD ["./moderation-server"]

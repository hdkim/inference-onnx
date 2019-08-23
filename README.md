# inference-onnx

INIT

How to update proto after chaning inference.proto file
$ cd grpclib
$ python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/inference.proto
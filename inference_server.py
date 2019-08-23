# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import time
import logging

import grpc

import grpclib.inference_pb2 as inference_pb2
import grpclib.inference_pb2_grpc as inference_pb2_grpc

import inferences.resnet.inferencer as resnet_inferencer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

inferencers = {'resnet': resnet_inferencer.inference}


class Inferencer(inference_pb2_grpc.InferencerServicer):

    def Inference(self, request, context):
        model = request.model
        data = request.data
        res = inferencers[model](data)
        print("request for model:", model, ", res:", res)
        return inference_pb2.InferenceResponse(result=res)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferencerServicer_to_server(Inferencer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()

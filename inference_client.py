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
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function
import logging

import numpy as np    # we're going to use numpy to process input and output data
import grpc
import time

import grpclib.inference_pb2 as inference_pb2
import grpclib.inference_pb2_grpc as inference_pb2_grpc

import inferences.util as util


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = inference_pb2_grpc.InferencerStub(channel)
        binary = util.image_to_byte_array('inferences/resnet/images/dog.jpg')
        start = time.time()
        response = stub.Inference(inference_pb2.InferenceRequest(model='resnet', data=binary))
        end = time.time()
        inference_time = np.round((end - start) * 1000, 2)
    print("Inference Time: ", str(inference_time) + "ms")
    print("client received: " + response.result)


if __name__ == '__main__':
    logging.basicConfig()
    run()

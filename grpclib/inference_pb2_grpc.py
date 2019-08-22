# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import inference_pb2 as inference__pb2


class InferencerStub(object):
  """The inference service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Inference = channel.unary_unary(
        '/inference.Inferencer/Inference',
        request_serializer=inference__pb2.InferenceRequest.SerializeToString,
        response_deserializer=inference__pb2.InferenceResponse.FromString,
        )


class InferencerServicer(object):
  """The inference service definition.
  """

  def Inference(self, request, context):
    """Sends a greeting
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_InferencerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Inference': grpc.unary_unary_rpc_method_handler(
          servicer.Inference,
          request_deserializer=inference__pb2.InferenceRequest.FromString,
          response_serializer=inference__pb2.InferenceResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'inference.Inferencer', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import grpc
import numpy as np
import torch

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.utils import create_custom_module

_IDLE_FISH_REUSE_EMBEDDING_GRPC_CHANNEL = os.environ.get(
    "IDLE_FISH_ENABLE_EMBEDDING_GRPC_CHANNEL_REUSE", ""
).strip() in ("1", "true", "True")


def tensor_pb_to_torch(tensor_pb) -> Optional[torch.Tensor]:
    data_type = tensor_pb.data_type
    shape = list(tensor_pb.shape)
    if data_type == tensor_pb.DataType.FP32:
        dtype = torch.float32
        buffer = tensor_pb.fp32_data
    elif data_type == tensor_pb.DataType.INT32:
        dtype = torch.int32
        buffer = tensor_pb.int32_data
    elif data_type == tensor_pb.DataType.FP16:
        dtype = torch.float16
        buffer = tensor_pb.fp16_data
    elif data_type == tensor_pb.DataType.BF16:
        dtype = torch.bfloat16
        buffer = tensor_pb.bf16_data
    else:
        logging.warning(f"Unsupported data type: {data_type}")
        return None
    if not buffer:
        logging.warning("Empty data buffer")
        return None
    try:
        torch_tensor = torch.frombuffer(buffer, dtype=dtype).clone().reshape(shape)
        return torch_tensor
    except Exception as e:
        logging.warning(f"Failed to convert buffer to tensor: {e}")
        return None


class EmbeddingEndpoint(object):
    def __init__(
        self,
        model_config,
        grpc_config,
        server_config,
        tokenizer: BaseTokenizer,
    ):
        self.renderer = create_custom_module(model_config, tokenizer).renderer
        # 创建到服务器的连接
        self.address = f"localhost:{server_config.embedding_rpc_server_port}"
        logging.info(f"embedding endpoint connect to rpc addresses: {self.address}")
        self.options = []
        client_config = grpc_config.get_client_config()
        if client_config is not None:
            for key, value in client_config.items():
                self.options.append((key, value))
        logging.info(f"embedding endpoint grpc options: {self.options}")
        self.reuse_grpc_channel = _IDLE_FISH_REUSE_EMBEDDING_GRPC_CHANNEL
        self._channel = None
        self._stub = None
        self._channel_lock = asyncio.Lock()
        if self.reuse_grpc_channel:
            logging.info("idlefish embedding endpoint grpc channel reuse enabled")

    async def _get_embedding_stub(self):
        if self._stub is not None:
            return self._stub
        async with self._channel_lock:
            if self._stub is None:
                self._channel = grpc.aio.insecure_channel(
                    self.address, options=self.options
                )
                self._stub = pb2_grpc.EmbeddingRpcServiceStub(self._channel)
        return self._stub

    async def _reset_embedding_stub(self):
        async with self._channel_lock:
            channel = self._channel
            self._channel = None
            self._stub = None
        if channel is not None:
            await channel.close()

    async def embedding(
        self, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if isinstance(request, str):
            request = json.loads(request)
        try:
            formate_request = self.renderer.render_request(request)
            batch_input = self.renderer.create_input(formate_request)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))
        try:
            batch_output = await self.generate_embeddings(batch_input)
            response = await self.renderer.render_response(
                formate_request, batch_input, batch_output
            )
            logable_response = await self.renderer.render_log_response(response)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, str(e))
        return response, logable_response

    async def generate_embeddings(self, input: EngineInputs):
        output = EngineOutputs(outputs=None, input_length=0)
        await self.generate_embeddings_grpc(input, output)
        return output

    async def generate_embeddings_grpc(
        self, input: EngineInputs, output: EngineOutputs
    ):
        channel = None
        if self.reuse_grpc_channel:
            stub = await self._get_embedding_stub()
        else:
            channel = grpc.aio.insecure_channel(self.address, options=self.options)
            stub = pb2_grpc.EmbeddingRpcServiceStub(channel)
        multimodal_features = []
        for feature in input.multimodal_inputs:
            multimodal_features.append(
                pb2.MultimodalInputPB(
                    multimodal_type=feature.mm_type, multimodal_url=feature.url
                )
            )
        request = pb2.EmbeddingInputPB(
            token_ids=input.token_ids.tolist(),  # 示例token ids
            token_type_ids=input.token_type_ids.tolist(),  # 示例segment ids
            input_lengths=input.input_lengths.tolist(),  # 输入长度
            request_id=1,  # 唯一请求ID
            multimodal_features=multimodal_features,
        )
        try:
            try:
                response = await stub.embedding(request)
            except grpc.RpcError as e:
                if self.reuse_grpc_channel and e.code() == grpc.StatusCode.UNAVAILABLE:
                    logging.warning(
                        "embedding grpc channel unavailable, reset and retry once"
                    )
                    await self._reset_embedding_stub()
                    await asyncio.sleep(0.2)
                    stub = await self._get_embedding_stub()
                    response = await stub.embedding(request)
                else:
                    raise
            if response.output_is_tensor:
                tensor_pb = response.output_t
                result = tensor_pb_to_torch(tensor_pb)
            else:
                result = []
                for output_map_iter in response.output_map:
                    tensor_map = {}
                    for key, tensor_pb in output_map_iter.items():
                        torch_tensor = tensor_pb_to_torch(tensor_pb)
                        tensor_map[key] = torch_tensor
                    result.append(tensor_map)
            output.outputs = result
            output.input_length = input.input_length
            return result
        except grpc.RpcError as e:
            logging.warning(f"RPC failed: {e.code()}: {e.details()}")
            raise
        finally:
            if channel is not None:
                await channel.close()

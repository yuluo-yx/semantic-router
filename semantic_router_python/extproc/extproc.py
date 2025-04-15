"""
ExtProc service implementation for semantic router.
This is based on  https://github.com/rootfs/production-stack/tree/extproc-cache
"""

import json
import logging
import signal
import sys
import threading
from concurrent import futures
from typing import Dict, List, Optional

import grpc
from google.protobuf.any_pb2 import Any

# Try to import Envoy ExtProc protos
try:
    from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_proc_grpc
    from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
    extproc_available = True
except ImportError:
    extproc_available = False

from semantic_router_python.config import RouterConfig, load_config

logger = logging.getLogger(__name__)

# Global initialization lock
_init_lock = threading.Lock()
_initialized = False


class ChatMessage:
    """Represents a message in the OpenAI chat format."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class OpenAIRequest:
    """Represents an OpenAI API request."""
    
    def __init__(self, model: str, messages: List[ChatMessage]):
        self.model = model
        self.messages = messages


class ExtProcService(ext_proc_grpc.ExternalProcessorServicer):
    """An Envoy ExtProc server that routes OpenAI API requests."""
    
    def __init__(self, config_path: str):
        """Initialize the ExtProc service with the given config."""
        self.config = load_config(config_path)
        self.task_descriptions = self.config.get_task_descriptions()
        # TODO: Implement actual model initialization
        logger.info(f"Initializing BERT model: {self.config.bert_model.model_id}, use_cpu: {self.config.bert_model.use_cpu}")
    
    def Process(self, request_iterator, context):
        """Process ExtProc requests from Envoy."""
        logger.info("Started processing a new request")
        
        for request in request_iterator:
            if request.HasField("request_body"):
                body = request.request_body
                logger.info(f"Request body: {body}")
                
                # Parse request body and route based on semantic content
                try:
                    openai_request = self._parse_openai_request(body.body.as_bytes())
                    
                    # Extract user query from messages
                    user_query = self._extract_user_query(openai_request.messages)
                    
                    # TODO: Implement semantic routing logic
                    # This would use a Python equivalent of the semantic matching in Go
                    
                    # Create and yield response
                    response = ext_proc_pb2.ProcessingResponse()
                    # Populate response with routing decision
                    
                    yield response
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(str(e))
                    return
            else:
                logger.warning(f"Unknown request type: {request}")
    
    def _parse_openai_request(self, data: bytes) -> OpenAIRequest:
        """Parse the OpenAI request JSON."""
        try:
            request_dict = json.loads(data)
            messages = []
            for msg in request_dict.get("messages", []):
                messages.append(ChatMessage(
                    role=msg.get("role", ""),
                    content=msg.get("content", "")
                ))
            return OpenAIRequest(
                model=request_dict.get("model", ""),
                messages=messages
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI request: {e}")
            raise
    
    def _extract_user_query(self, messages: List[ChatMessage]) -> str:
        """Extract the user query from the last user message."""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""
    
    def _extract_roles_from_messages(self, messages: List[ChatMessage]) -> str:
        """Extract roles from chat messages."""
        roles = [msg.role for msg in messages if msg.role]
        return "\n".join(roles)


def serve_extproc(service: ExtProcService, port: int, grace_period: int = 5):
    """Start the ExtProc server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ext_proc_grpc.add_ExternalProcessorServicer_to_server(service, server)
    
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Server started on port {port}")
    
    def handle_shutdown(signum, frame):
        logger.info("Received shutdown signal, gracefully stopping server...")
        stopped_event = threading.Event()
        
        def stop_server():
            server.stop(grace_period)
            stopped_event.set()
            logger.info("Server stopped")
        
        stop_thread = threading.Thread(target=stop_server)
        stop_thread.daemon = True
        stop_thread.start()
        
        try:
            stopped_event.wait(timeout=grace_period + 1)
        except:
            logger.warning("Server stop timed out, forcing exit")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Keep thread alive
    try:
        while True:
            signal.pause()
    except (KeyboardInterrupt, SystemExit):
        pass 
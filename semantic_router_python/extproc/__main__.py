"""
Entrypoint for the semantic router ExtProc service.
"""

import argparse
import logging
import sys

from semantic_router_python.extproc import ExtProcService, serve_extproc

try:
    from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
    extproc_available = True
except ImportError:
    extproc_available = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Semantic Router ExtProc Service")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/router.json",
        help="Path to the router configuration file"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--extproc-port", 
        type=int, 
        default=50051,
        help="Port for the ExtProc gRPC service"
    )
    parser.add_argument(
        "--extproc-grace-period", 
        type=int, 
        default=5,
        help="Grace period in seconds for server shutdown"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the ExtProc service."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("semantic_router.extproc")
    
    # Check if ExtProc is available
    if not extproc_available:
        logger.error(
            "Envoy ExtProc protobuf definitions not available. "
            "Please install the required dependencies."
        )
        sys.exit(1)
    
    # Create and start the service
    try:
        logger.info(f"Creating ExtProc service with config: {args.config}")
        service = ExtProcService(args.config)
        
        logger.info(f"Starting ExtProc service on port {args.extproc_port}...")
        serve_extproc(
            service=service,
            port=args.extproc_port,
            grace_period=args.extproc_grace_period
        )
    except Exception as e:
        logger.error(f"Failed to start ExtProc service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
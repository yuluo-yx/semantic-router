package main

import (
	"flag"
	"log"
	"os"

	"github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/extproc"
)

func main() {
	// Parse command-line flags
	var (
		configPath = flag.String("config", "config/config.yaml", "Path to the configuration file")
		port       = flag.Int("port", 50051, "Port to listen on")
	)
	flag.Parse()

	// Check if config file exists
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		log.Fatalf("Config file not found: %s", *configPath)
	}

	// Create and start the server
	server, err := extproc.NewServer(*configPath, *port)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	log.Printf("Starting LLM Semantic Router ExtProc with config: %s", *configPath)
	if err := server.Start(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

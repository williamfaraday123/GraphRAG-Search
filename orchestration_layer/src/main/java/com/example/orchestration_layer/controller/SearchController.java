package com.example.orchestration_layer.controller;

import com.example.orchestration_layer.grpc.GrpcQueryClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;

/**
 * REST controller that bridges HTTP POST → gRPC → SSE stream to the React UI.
 *
 * Flow (matches diagrams/sequence.plantuml):
 *   HTTP POST /api/v1/search {prompt: "..."}
 *     → gRPC StartProcessing(prompt)
 *       → SSE: STATUS events (RETRIEVING → ITERATING → SYNTHESIZING)
 *       → SSE: CHUNK events (streaming tokens)
 *       → SSE: DONE  event  (final answer + sources + confidence)
 */
@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*") // Allow React dev server on any port
public class SearchController {

    private static final Logger log = LoggerFactory.getLogger(SearchController.class);
    private final GrpcQueryClient grpcClient;

    public SearchController(GrpcQueryClient grpcClient) {
        this.grpcClient = grpcClient;
    }

    /**
     * POST /api/v1/search
     * Body: { "prompt": "what is machine learning?", "topK": 5 }
     * Returns: text/event-stream (SSE)
     */
    @PostMapping(value = "/search", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter search(@RequestBody Map<String, Object> body) {
        String prompt = (String) body.getOrDefault("prompt", "");
        int topK = (int) body.getOrDefault("topK", 5);

        SseEmitter emitter = new SseEmitter(300_000L); // 5-minute timeout
        log.info("Search request received: prompt='{}', topK={}", prompt, topK);

        grpcClient.startSearch(prompt, topK, event -> {
            try {
                switch (event.getType()) {
                    case STATUS -> emitter.send(SseEmitter.event()
                            .name("status")
                            .data(Map.of(
                                    "stage", event.getStage(),
                                    "message", event.getMessage()
                            )));
                    case CHUNK -> emitter.send(SseEmitter.event()
                            .name("chunk")
                            .data(Map.of(
                                    "content", event.getContent(),
                                    "sourceId", event.getSourceId() != null ? event.getSourceId() : ""
                            )));
                    case DONE -> {
                        emitter.send(SseEmitter.event()
                                .name("done")
                                .data(Map.of(
                                        "answer", event.getAnswer(),
                                        "sources", event.getSources(),
                                        "confidence", event.getConfidence()
                                )));
                        emitter.complete();
                    }
                    case ERROR -> {
                        emitter.send(SseEmitter.event()
                                .name("error")
                                .data(Map.of("message", event.getMessage())));
                        emitter.completeWithError(new RuntimeException(event.getMessage()));
                    }
                }
            } catch (IOException ex) {
                log.error("SSE send error: {}", ex.getMessage());
                emitter.completeWithError(ex);
            }
        });

        emitter.onCompletion(() -> log.info("SSE stream completed for: {}", prompt));
        emitter.onTimeout(() -> log.warn("SSE stream timed out for: {}", prompt));
        emitter.onError(ex -> log.error("SSE stream error: {}", ex.getMessage()));

        return emitter;
    }

    /** Health-check endpoint */
    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "UP", "service", "orchestration-layer");
    }
}

package com.example.orchestration_layer.controller;

import com.example.orchestration_layer.grpc.GrpcQueryClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * REST controller that bridges HTTP POST → gRPC → SSE stream to the React UI.
 *
 * Flow (matches diagrams/sequence.plantuml):
 *   HTTP POST /api/v1/search {prompt: "..."}
 *     → gRPC StartProcessing(prompt)
 *       → SSE: STATUS events (PLANNING → RETRIEVING → ANALYZING → SYNTHESIZING)
 *       → SSE: CHUNK events (streaming tokens)
 *       → SSE: DONE  event  (final answer + sources + confidence)
 */
@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*") // Allow React dev server on any port
public class SearchController {

    private static final Logger log = LoggerFactory.getLogger(SearchController.class);
    private final GrpcQueryClient grpcClient;
    private final long sseTimeoutMs;

    public SearchController(GrpcQueryClient grpcClient,
                            @Value("${search.sse-timeout-ms:900000}") long sseTimeoutMs) {
        this.grpcClient = grpcClient;
        this.sseTimeoutMs = sseTimeoutMs;
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
        String corpusId = body.get("corpusId") instanceof String value ? value : "wiki";
        List<String> fileIds = new ArrayList<>();
        if (body.get("fileIds") instanceof List<?> ids) {
            for (Object id : ids) {
                if (id instanceof String s && !s.isBlank()) fileIds.add(s);
            }
        }
        List<GrpcQueryClient.Turn> history = parseHistory(body.get("history"));

        SseEmitter emitter = new SseEmitter(sseTimeoutMs);
        log.info("Search request received: prompt='{}', topK={}, corpusId={}, fileIds={}, earlierTurns={}",
                prompt, topK, corpusId, fileIds.isEmpty() ? "all" : fileIds, history.size());

        grpcClient.startSearch(prompt, topK, corpusId, fileIds, history, event -> {
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
                    case AGENT_STEP -> emitter.send(SseEmitter.event()
                            .name("agent_step")
                            .data(Map.of(
                                    "agent", event.getAgentName(),
                                    "detail", event.getAgentDetail()
                            )));
                    case DONE -> {
                        emitter.send(SseEmitter.event()
                                .name("done")
                                .data(Map.of(
                                        "answer", event.getAnswer(),
                                        "sources", event.getSources(),
                                        "confidence", event.getConfidence(),
                                        "chunks", event.getChunks(),
                                        // Knowledge-graph fact paths behind the answer, and where the walk started
                                        "graphPaths", event.getGraphPaths() == null ? List.of() : event.getGraphPaths(),
                                        "graphSeeds", event.getGraphSeeds() == null ? List.of() : event.getGraphSeeds(),
                                        // Sent back with the next question as part of "history"
                                        "entityKeys", event.getEntityKeys() == null ? List.of() : event.getEntityKeys(),
                                        "standaloneQuery", event.getStandaloneQuery() == null ? "" : event.getStandaloneQuery()
                                )));
                        emitter.complete();
                    }
                    case ERROR -> {
                        emitter.send(SseEmitter.event()
                                .name("error")
                                .data(Map.of("message", event.getMessage())));
                        emitter.complete();
                    }
                }
            } catch (IOException ex) {
                log.error("SSE send error: {}", ex.getMessage());
                emitter.completeWithError(ex);
            }
        });

        emitter.onCompletion(() -> log.info("SSE stream completed for: {}", prompt));
        emitter.onTimeout(() -> {
            log.warn("SSE stream timed out after {} ms for: {}", sseTimeoutMs, prompt);
            // Tell the UI instead of silently closing, so it doesn't look like it's still running.
            try {
                emitter.send(SseEmitter.event().name("error").data(Map.of(
                        "message", "Search timed out after " + (sseTimeoutMs / 60000)
                                + " minutes. The LLM may be running on CPU; try a shorter question or enable GPU.")));
            } catch (IOException | IllegalStateException ignored) {
                // Stream already closed
            }
            emitter.complete();
        });
        emitter.onError(ex -> log.error("SSE stream error: {}", ex.getMessage()));

        return emitter;
    }

    /** Most earlier turns forwarded; the query processor uses only the last three. */
    private static final int MAX_HISTORY_TURNS = 3;
    private static final int MAX_TURN_TEXT = 600;
    private static final int MAX_TURN_KEYS = 8;

    /**
     * "history": [{"question": "...", "answer": "...", "entityKeys": ["..."]}, ...], oldest first.
     * Malformed entries are skipped; text is capped so a client can't send whole documents.
     */
    static List<GrpcQueryClient.Turn> parseHistory(Object raw) {
        List<GrpcQueryClient.Turn> turns = new ArrayList<>();
        if (!(raw instanceof List<?> items)) return turns;
        for (Object item : items) {
            if (!(item instanceof Map<?, ?> turn)) continue;
            String question = turn.get("question") instanceof String q ? q.strip() : "";
            if (question.isEmpty()) continue;
            String answer = turn.get("answer") instanceof String a ? a.strip() : "";
            List<String> keys = new ArrayList<>();
            if (turn.get("entityKeys") instanceof List<?> ks) {
                for (Object k : ks) {
                    if (k instanceof String s && !s.isBlank() && keys.size() < MAX_TURN_KEYS) keys.add(s);
                }
            }
            turns.add(new GrpcQueryClient.Turn(cap(question), cap(answer), keys));
        }
        return turns.size() > MAX_HISTORY_TURNS
                ? new ArrayList<>(turns.subList(turns.size() - MAX_HISTORY_TURNS, turns.size()))
                : turns;
    }

    private static String cap(String text) {
        return text.length() > MAX_TURN_TEXT ? text.substring(0, MAX_TURN_TEXT) : text;
    }

    /**
     * GET /api/v1/chunk-source?uri=s3://rag-datasets/file.txt
     * Proxies a full source-document fetch through the Query Processor so the
     * browser never needs direct access to MinIO/object storage.
     */
    @GetMapping("/chunk-source")
    public Map<String, Object> chunkSource(@RequestParam String uri) {
        try {
            String content = grpcClient.getChunkSource(uri);
            return Map.of("found", true, "content", content);
        } catch (Exception ex) {
            log.warn("chunk-source fetch failed for '{}': {}", uri, ex.getMessage());
            return Map.of("found", false, "error", ex.getMessage() != null ? ex.getMessage() : "Failed to fetch source");
        }
    }

    /** Health-check endpoint */
    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "UP", "service", "orchestration-layer");
    }
}

package com.example.orchestration_layer.controller;

import java.util.List;
import java.util.Map;

/**
 * SSE event payload sent from the server to the React UI.
 */
public class SearchEvent {

    public enum EventType {
        STATUS, CHUNK, DONE, ERROR, AGENT_STEP
    }

    private EventType type;
    private String stage;        // for STATUS: RETRIEVING, ITERATING, SYNTHESIZING
    private String message;       // for STATUS / ERROR
    private String content;       // for CHUNK (a word or token)
    private String sourceId;      // for CHUNK source metadata
    private String answer;        // for DONE: final full answer
    private List<String> sources; // for DONE: citation sources
    private double confidence;    // for DONE: confidence score
    private List<Map<String, Object>> chunks; // for DONE: ranked evidence chunks
    private List<Map<String, Object>> graphPaths; // for DONE: knowledge-graph fact paths behind the answer
    private List<String> graphSeeds;  // for DONE: entities named in the question that the graph walk started from
    private String agentName;     // for AGENT_STEP: which agent acted (planner/retriever/analyst/synthesizer)
    private String agentDetail;   // for AGENT_STEP: what it did/decided

    public SearchEvent() {}

    // --- Factory methods ---
    public static SearchEvent status(String stage, String message) {
        var e = new SearchEvent();
        e.type = EventType.STATUS;
        e.stage = stage;
        e.message = message;
        return e;
    }

    public static SearchEvent chunk(String content, String sourceId) {
        var e = new SearchEvent();
        e.type = EventType.CHUNK;
        e.content = content;
        e.sourceId = sourceId;
        return e;
    }

    public static SearchEvent done(String answer, List<String> sources, double confidence,
                                    List<Map<String, Object>> chunks) {
        var e = new SearchEvent();
        e.type = EventType.DONE;
        e.answer = answer;
        e.sources = sources;
        e.confidence = confidence;
        e.chunks = chunks;
        return e;
    }

    public static SearchEvent error(String message) {
        var e = new SearchEvent();
        e.type = EventType.ERROR;
        e.message = message;
        return e;
    }

    public static SearchEvent agentStep(String agentName, String detail) {
        var e = new SearchEvent();
        e.type = EventType.AGENT_STEP;
        e.agentName = agentName;
        e.agentDetail = detail;
        return e;
    }

    // --- Getters / Setters ---
    public EventType getType() { return type; }
    public void setType(EventType type) { this.type = type; }
    public String getStage() { return stage; }
    public void setStage(String stage) { this.stage = stage; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public String getSourceId() { return sourceId; }
    public void setSourceId(String sourceId) { this.sourceId = sourceId; }
    public String getAnswer() { return answer; }
    public void setAnswer(String answer) { this.answer = answer; }
    public List<String> getSources() { return sources; }
    public void setSources(List<String> sources) { this.sources = sources; }
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    public List<Map<String, Object>> getChunks() { return chunks; }
    public void setChunks(List<Map<String, Object>> chunks) { this.chunks = chunks; }
    public List<Map<String, Object>> getGraphPaths() { return graphPaths; }
    public void setGraphPaths(List<Map<String, Object>> graphPaths) { this.graphPaths = graphPaths; }
    public List<String> getGraphSeeds() { return graphSeeds; }
    public void setGraphSeeds(List<String> graphSeeds) { this.graphSeeds = graphSeeds; }
    public String getAgentName() { return agentName; }
    public void setAgentName(String agentName) { this.agentName = agentName; }
    public String getAgentDetail() { return agentDetail; }
    public void setAgentDetail(String agentDetail) { this.agentDetail = agentDetail; }
}

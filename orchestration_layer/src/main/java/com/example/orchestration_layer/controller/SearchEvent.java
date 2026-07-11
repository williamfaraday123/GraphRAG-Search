package com.example.orchestration_layer.controller;

import java.util.List;

/**
 * SSE event payload sent from the server to the React UI.
 */
public class SearchEvent {

    public enum EventType {
        STATUS, CHUNK, DONE, ERROR
    }

    private EventType type;
    private String stage;        // for STATUS: RETRIEVING, ITERATING, SYNTHESIZING
    private String message;       // for STATUS / ERROR
    private String content;       // for CHUNK (a word or token)
    private String sourceId;      // for CHUNK source metadata
    private String answer;        // for DONE: final full answer
    private List<String> sources; // for DONE: citation sources
    private double confidence;    // for DONE: confidence score

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

    public static SearchEvent done(String answer, List<String> sources, double confidence) {
        var e = new SearchEvent();
        e.type = EventType.DONE;
        e.answer = answer;
        e.sources = sources;
        e.confidence = confidence;
        return e;
    }

    public static SearchEvent error(String message) {
        var e = new SearchEvent();
        e.type = EventType.ERROR;
        e.message = message;
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
}

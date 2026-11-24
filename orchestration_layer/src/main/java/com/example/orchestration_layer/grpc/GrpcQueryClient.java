package com.example.orchestration_layer.grpc;

import com.example.orchestration_layer.controller.SearchEvent;
import com.example.orchestration_layer.controller.SearchEvent.EventType;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import com.google.protobuf.ByteString;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * gRPC client that connects to the Python Query Processor.
 * Bridges the gRPC stream into SSE-friendly events for the REST controller.
 */
@Service
public class GrpcQueryClient {

    private static final Logger log = LoggerFactory.getLogger(GrpcQueryClient.class);

    private final String grpcHost;
    private final int grpcPort;
    private io.grpc.ManagedChannel channel;
    private QueryProcessorServiceGrpc.QueryProcessorServiceStub asyncStub;
    private QueryProcessorServiceGrpc.QueryProcessorServiceBlockingStub blockingStub;

    public GrpcQueryClient(
            @Value("${grpc.client.host:localhost}") String grpcHost,
            @Value("${grpc.client.port:50051}") int grpcPort) {
        this.grpcHost = grpcHost;
        this.grpcPort = grpcPort;
    }

    @PostConstruct
    public void init() {
        channel = io.grpc.ManagedChannelBuilder
                .forAddress(grpcHost, grpcPort)
                .usePlaintext()
                .build();
        asyncStub = QueryProcessorServiceGrpc.newStub(channel);
        blockingStub = QueryProcessorServiceGrpc.newBlockingStub(channel);
        log.info("gRPC client connected to {}:{}", grpcHost, grpcPort);
    }

    @PreDestroy
    public void shutdown() throws InterruptedException {
        if (channel != null) {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    /**
     * Starts the search pipeline on the Query Processor and streams
     * results through the provided event consumer.
     */
    /** One earlier question of the chat, its (trimmed) answer, and the entity keys it named. */
    public record Turn(String question, String answer, List<String> entityKeys) {}

    public void startSearch(String prompt, int topK, String corpusId, List<String> fileIds,
                            List<Turn> history, Consumer<SearchEvent> eventConsumer) {
        var builder = QueryProto.SearchRequest.newBuilder()
                .setPrompt(prompt)
                .setTopK(topK)
                .setUseGraphRefinement(true)
                .setCorpusId(corpusId)
                .addAllFileIds(fileIds); // empty = search every file in the corpus
        for (Turn turn : history) { // empty = first question of the chat
            builder.addHistory(QueryProto.ConversationTurn.newBuilder()
                    .setQuestion(turn.question())
                    .setAnswer(turn.answer())
                    .addAllEntityKeys(turn.entityKeys()));
        }
        var request = builder.build();

        asyncStub.startProcessing(request, new StreamObserver<QueryProto.SearchResponse>() {
            @Override
            public void onNext(QueryProto.SearchResponse response) {
                if (response.hasStatus()) {
                    var s = response.getStatus();
                    eventConsumer.accept(SearchEvent.status(s.getStage(), s.getMessage()));
                } else if (response.hasChunk()) {
                    var c = response.getChunk();
                    eventConsumer.accept(SearchEvent.chunk(c.getContent(), c.getSourceId()));
                } else if (response.hasAnswer()) {
                    var a = response.getAnswer();
                    List<Map<String, Object>> chunks = new ArrayList<>();
                    for (var c : a.getChunksList()) {
                        Map<String, Object> chunk = new LinkedHashMap<>();
                        chunk.put("chunkId", c.getChunkId());
                        chunk.put("content", c.getContent());
                        chunk.put("source", c.getSource());
                        chunk.put("milvusScore", c.getMilvusScore());
                        chunk.put("pagerankScore", c.getPagerankScore());
                        chunk.put("graphScore", c.getGraphScore());
                        chunk.put("foundByGraph", c.getFoundByGraph());
                        chunk.put("hybridScore", c.getHybridScore());
                        chunks.add(chunk);
                    }
                    List<Map<String, Object>> graphPaths = new ArrayList<>();
                    for (var p : a.getGraphPathsList()) {
                        List<Map<String, Object>> facts = new ArrayList<>();
                        for (var f : p.getFactsList()) {
                            Map<String, Object> fact = new LinkedHashMap<>();
                            fact.put("source", f.getSource());
                            fact.put("sourceLabel", f.getSourceLabel());
                            fact.put("relation", f.getRelation());
                            fact.put("target", f.getTarget());
                            fact.put("targetLabel", f.getTargetLabel());
                            fact.put("chunkId", f.getChunkId());
                            fact.put("fileId", f.getFileId());
                            fact.put("filename", f.getFilename());
                            fact.put("evidence", f.getEvidence());
                            facts.add(fact);
                        }
                        graphPaths.add(Map.of(
                                "score", p.getScore(),
                                "kind", p.getKind().isEmpty() ? "reasoning" : p.getKind(),
                                "facts", facts));
                    }
                    var done = SearchEvent.done(
                            a.getText(),
                            a.getSourcesList(),
                            a.getConfidenceScore(),
                            chunks
                    );
                    done.setGraphPaths(graphPaths);
                    done.setGraphSeeds(a.getGraphSeedsList());
                    done.setEntityKeys(a.getEntityKeysList());
                    done.setStandaloneQuery(a.getStandaloneQuery());
                    eventConsumer.accept(done);
                } else if (response.hasAgentStep()) {
                    var s = response.getAgentStep();
                    eventConsumer.accept(SearchEvent.agentStep(s.getAgent(), s.getDetail()));
                }
            }

            @Override
            public void onError(Throwable t) {
                log.error("gRPC stream error: {}", t.getMessage());
                eventConsumer.accept(SearchEvent.error(t.getMessage()));
            }

            @Override
            public void onCompleted() {
                log.info("gRPC stream completed");
            }
        });
    }

    /**
     * Fetches the full source document for one evidence chunk from object storage,
     * via the Query Processor — the browser never talks to MinIO directly.
     */
    public String getChunkSource(String source) {
        var request = QueryProto.ChunkSourceRequest.newBuilder()
                .setSource(source)
                .build();

        var response = blockingStub
                .withDeadlineAfter(15, TimeUnit.SECONDS)
                .getChunkSource(request);

        if (!response.getFound()) {
            throw new RuntimeException(
                    response.getError().isEmpty() ? "Source not found" : response.getError());
        }
        return response.getContent();
    }

    public QueryProto.UploadCorpusFileResponse uploadCorpusFile(
            String corpusId, String filename, byte[] content) {
        var request = QueryProto.UploadCorpusFileRequest.newBuilder()
                .setCorpusId(corpusId)
                .setFilename(filename)
                .setContent(ByteString.copyFrom(content))
                .build();
        return blockingStub.uploadCorpusFile(request);
    }

    public List<Map<String, String>> listCorpusFiles(String corpusId) {
        var request = QueryProto.ListCorpusFilesRequest.newBuilder()
                .setCorpusId(corpusId)
                .build();
        var response = blockingStub.listCorpusFiles(request);
        List<Map<String, String>> files = new ArrayList<>();
        for (var file : response.getFilesList()) {
            files.add(Map.of("fileId", file.getFileId(), "filename", file.getFilename()));
        }
        return files;
    }

    public QueryProto.GetCorpusFileResponse getCorpusFile(String corpusId, String fileId) {
        return blockingStub
                .withDeadlineAfter(15, TimeUnit.SECONDS)
                .getCorpusFile(corpusFileRequest(corpusId, fileId));
    }

    public QueryProto.DeleteCorpusFileResponse deleteCorpusFile(String corpusId, String fileId) {
        return blockingStub
                .withDeadlineAfter(30, TimeUnit.SECONDS)
                .deleteCorpusFile(corpusFileRequest(corpusId, fileId));
    }

    /** Builds the corpus map; embedding every chunk on CPU can take a while for large workspaces. */
    public QueryProto.CorpusMapResponse getCorpusMap(String corpusId, List<String> fileIds) {
        return blockingStub
                .withDeadlineAfter(120, TimeUnit.SECONDS)
                .getCorpusMap(QueryProto.CorpusMapRequest.newBuilder()
                        .setCorpusId(corpusId)
                        .addAllFileIds(fileIds)
                        .build());
    }

    private static QueryProto.CorpusFileRequest corpusFileRequest(String corpusId, String fileId) {
        return QueryProto.CorpusFileRequest.newBuilder()
                .setCorpusId(corpusId)
                .setFileId(fileId)
                .build();
    }
}

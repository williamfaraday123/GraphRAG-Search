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
    public void startSearch(String prompt, int topK, Consumer<SearchEvent> eventConsumer) {
        var request = QueryProto.SearchRequest.newBuilder()
                .setPrompt(prompt)
                .setTopK(topK)
                .setUseGraphRefinement(true)
                .build();

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
                    eventConsumer.accept(SearchEvent.done(
                            a.getText(),
                            a.getSourcesList(),
                            a.getConfidenceScore()
                    ));
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
}

# ============================================
# Orchestration Layer — Dockerfile
# Build JAR first: mvnw package -DskipTests
# ============================================

FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

COPY target/*.jar app.jar

EXPOSE 8080

ENV GRPC_CLIENT_HOST=query-processor
ENV GRPC_CLIENT_PORT=50051

ENTRYPOINT ["java", "-jar", "app.jar"]

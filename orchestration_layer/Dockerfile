# ============================================
# Orchestration Layer — Multi-stage Dockerfile
# ============================================

# ---- Stage 1: Build React UI ----
FROM node:22-alpine AS react-builder

WORKDIR /ui
COPY web_search_client/package*.json ./
RUN npm ci
COPY web_search_client/ ./
RUN npm run build

# ---- Stage 2: Build Spring Boot ----
FROM maven:3.9.9-eclipse-temurin-25-alpine AS java-builder

WORKDIR /build

# Copy pom first (layer caching)
COPY pom.xml ./
COPY .mvn .mvn
RUN mkdir -p src/main/java src/main/resources src/main/proto

# Copy source, proto, and resources
COPY src src

# Copy built React app into Spring Boot's static resources
COPY --from=react-builder /ui/build src/main/resources/static/

# Generate proto stubs and package fat JAR
RUN --mount=type=cache,target=/root/.m2 \
    mvn package -DskipTests -Djar

# ---- Stage 3: Run ----
FROM eclipse-temurin:25-jre-alpine

WORKDIR /app

COPY --from=java-builder /build/target/*.jar app.jar

EXPOSE 8080

ENV SPRING_PROFILES_ACTIVE=prod
ENV GRPC_CLIENT_HOST=query_processor
ENV GRPC_CLIENT_PORT=50051

ENTRYPOINT ["java", "-jar", "app.jar"]

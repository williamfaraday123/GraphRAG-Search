### 3. Generate proto stubs & run Spring Boot

In a **new terminal**:

```powershell
cd orchestration_layer
mvnw compile -DskipTests
mvnw spring-boot:run
```

But first, the proto stubs need to be generated. You can do it in one shot with Maven's compile phase:

```powershell
cd orchestration_layer

# Step 1: Generate gRPC stubs from proto + compile
mvnw compile

# Step 2: If compile succeeds, run it
mvnw spring-boot:run
```